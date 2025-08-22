"""A method for fair representations."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Self

from flax import nnx
from flax_typed import jit, value_and_grad
from jax import Array
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import optax  # type: ignore[import]
from sklearn.base import BaseEstimator

from fair_forge.methods import FairnessType
from fair_forge.nn.utils import grad_reverse, iterate_forever
from fair_forge.preprocessing.definitions import GroupBasedTransform
from fair_forge.utils import batched

__all__ = ["Beutel"]


class Block(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        out = self.linear(x)
        out = nnx.sigmoid(out)
        return out


class Encoder(nnx.Module):
    def __init__(self, enc_size: Sequence[int], init_size: int, *, rngs: nnx.Rngs):
        layers: list[Block] = []
        if not enc_size:  # In the case that encoder size [] is specified
            layers.append(Block(init_size, init_size, rngs=rngs))
        else:
            layers.append(Block(init_size, enc_size[0], rngs=rngs))
            for k in range(len(enc_size) - 1):
                layers.append(Block(enc_size[k], enc_size[k + 1], rngs=rngs))
        self.encoder = nnx.Sequential(*layers)

    def __call__(self, x: Array) -> Array:
        return self.encoder(x)


class Adversary(nnx.Module):
    """Adversary of the GAN."""

    def __init__(
        self,
        fairness: FairnessType,
        adv_size: Sequence[int],
        init_size: int,
        s_size: int,
        adv_weight: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.fairness = fairness
        self.init_size = init_size
        self.adv_weight = adv_weight
        layers: list[Block | nnx.Linear] = []
        if not adv_size:  # In the case that encoder size [] is specified
            layers.append(nnx.Linear(init_size, s_size, rngs=rngs))
        else:
            layers.append(Block(init_size, adv_size[0], rngs=rngs))
            for k in range(len(adv_size) - 1):
                layers.append(Block(adv_size[k], adv_size[k + 1], rngs=rngs))
            layers.append(nnx.Linear(adv_size[-1], s_size, rngs=rngs))
        self.adversary = nnx.Sequential(*layers)

    def __call__(self, x: Array) -> Array:
        x = grad_reverse(x, lambda_=self.adv_weight)
        x = self.adversary(x)
        return x


class Predictor(nnx.Module):
    """Predictor of the GAN."""

    def __init__(
        self,
        pred_size: Sequence[int],
        init_size: int,
        class_label_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        layers: list[Block | nnx.Linear] = []
        if not pred_size:  # In the case that encoder size [] is specified
            layers.append(Block(init_size, class_label_size, rngs=rngs))
        else:
            layers.append(Block(init_size, pred_size[0], rngs=rngs))
            for k in range(len(pred_size) - 1):
                layers.append(Block(pred_size[k], pred_size[k + 1], rngs=rngs))
            layers.append(nnx.Linear(pred_size[-1], class_label_size, rngs=rngs))
        self.predictor = nnx.Sequential(*layers)

    def __call__(self, x: Array) -> Array:
        return self.predictor(x)


class Model(nnx.Module):
    """Whole GAN model."""

    def __init__(
        self,
        enc_size: Sequence[int],
        adv_size: Sequence[int],
        pred_size: Sequence[int],
        adv_weight: float,
        fairness: FairnessType,
        x_size: int,
        s_size: int,
        y_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.enc = Encoder(
            enc_size=enc_size,
            init_size=x_size,
            rngs=rngs,
        )
        self.adv = Adversary(
            fairness=fairness,
            adv_size=adv_size,
            init_size=enc_size[-1] if enc_size else x_size,
            s_size=s_size,
            adv_weight=adv_weight,
            rngs=rngs,
        )
        self.pred = Predictor(
            pred_size=pred_size,
            init_size=enc_size[-1] if enc_size else x_size,
            class_label_size=y_size,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        encoded = self.enc(x)
        s_hat = self.adv(encoded)
        y_hat = self.pred(encoded)
        return encoded, s_hat, y_hat


@dataclass
class Beutel(BaseEstimator, GroupBasedTransform):
    enc_size: list[int] = field(default_factory=lambda: [40])
    adv_size: list[int] = field(default_factory=lambda: [40])
    pred_size: list[int] = field(default_factory=lambda: [40])
    adv_weight: float = 1.0
    fairness: FairnessType = "dp"
    batch_size: int = 64
    iters: int = 500
    random_state: int = 42
    learning_rate: float = 0.005
    momentum: float = 0.9

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self:
        x_size = X.shape[1]
        y_size = classes if (classes := len(np.unique(y))) > 2 else 1
        s_size = n_groups if (n_groups := len(np.unique(groups))) > 2 else 1

        def loss_fn(model: Model, x: Array, y: Array, s: Array) -> Array:
            _, s_hat, y_hat = model(x)
            s_hat = s_hat.squeeze(-1)
            y_hat = y_hat.squeeze(-1)
            if y_size > 1:
                predictor_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=y_hat, labels=y
                ).mean()
            else:
                predictor_loss = optax.sigmoid_binary_cross_entropy(
                    logits=y_hat, labels=y
                ).mean()

            match self.fairness:
                case "eq_opp":
                    mask = y > 0.5
                case "eq_odds":
                    raise NotImplementedError("Not implemented Eq. Odds yet")
                case "dp":
                    mask = jnp.ones(s.shape, dtype=jnp.bool)
                case _:
                    raise ValueError("Invalid fairness value")
            if s_size > 1:
                adversary_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=s_hat, labels=s, where=mask
                ).mean()
            else:
                adversary_loss = optax.sigmoid_binary_cross_entropy(
                    logits=s_hat, labels=s
                ).mean()
            loss = predictor_loss + adversary_loss
            return loss

        @jit
        def train_step(
            model: Model,
            optimizer: nnx.Optimizer,
            metrics: nnx.MultiMetric,
            x: Array,
            y: Array,
            s: Array,
        ) -> None:
            """Train for a single step."""
            grad_fn = value_and_grad(loss_fn, has_aux=False)
            loss, grads = grad_fn(model, x, y, s)
            metrics.update(loss=loss)
            optimizer.update(grads)

        model = Model(
            enc_size=self.enc_size,
            adv_size=self.adv_size,
            pred_size=self.pred_size,
            adv_weight=self.adv_weight,
            fairness=self.fairness,
            x_size=x_size,
            y_size=y_size,
            s_size=s_size,
            rngs=nnx.Rngs(self.random_state),
        )
        optimizer = nnx.Optimizer(model, optax.adamw(self.learning_rate, self.momentum))
        metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        dataloader = iterate_forever(
            (jnp.asarray(X), jnp.asarray(y), jnp.asarray(groups)),
            batch_size=self.batch_size,
            seed=self.random_state,
        )
        for _, (X_batch, y_batch, groups_batch) in enumerate(
            islice(dataloader, self.iters)
        ):
            train_step(model, optimizer, metrics, X_batch, y_batch, groups_batch)

        self.enc_ = model.enc

        return self

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        x = jnp.asarray(X)
        transformed: list[Array] = []
        for slice_ in batched(len(x), self.batch_size, drop_last=False):
            X_batch = x[slice_]
            transformed.append(self.enc_(X_batch))
        return np.asarray(jnp.concat(transformed))

    def fit_transform(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> NDArray[np.float32]:
        self.fit(X, y, groups=groups)
        return self.transform(X)
