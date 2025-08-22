"""Protocols and implementations of methods for fairness-aware machine learning."""

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.metadata_routing import MetadataRequest

from fair_forge.utils import reproducible_random_state

__all__ = [
    "Blind",
    "FairnessType",
    "GroupMethod",
    "Majority",
    "Method",
    "Reweighting",
    "SampleWeightMethod",
]


class _MethodBase(Protocol):
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]: ...
    def get_params(self, deep: bool = ...) -> dict[str, object]: ...
    def set_params(self, **kwargs: Any) -> Self: ...
    def get_metadata_routing(self) -> MetadataRequest: ...


class Method(_MethodBase, Protocol):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> Self: ...


class SampleWeightMethod(_MethodBase, Protocol):
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        *,
        sample_weight: NDArray[np.float64],
    ) -> Self: ...


class GroupMethod(_MethodBase, Protocol):
    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self: ...


type FairnessType = Literal["dp", "eq_opp", "eq_odds"]


@dataclass
class Reweighting(BaseEstimator, GroupMethod):
    """An implementation of the Reweighing method from Kamiran&Calders, 2012.

    Args:
        base_method: The method to use for fitting and predicting. It should implement the
            SampleWeightMethod protocol.
    """

    base_method: SampleWeightMethod

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self:
        """Fit the model with reweighting based on group information."""
        # Verify that the input parameters all have the same length
        if not (len(X) == len(y) == len(groups)):
            raise ValueError(
                "X, y, and groups must all have the same length. "
                f"Got lengths {len(X)}, {len(y)}, and {len(groups)}."
            )
        sample_weight = _compute_instance_weights(y, groups=groups)
        self.base_method.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict using the fitted model."""
        return self.base_method.predict(X)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        params: dict[str, object] = {"name": self.base_method.__class__.__name__}
        if deep:
            params.update(self.base_method.get_params(deep=True))
        return {"base_method": params}


def _compute_instance_weights(
    y: NDArray[np.int32],
    *,
    groups: NDArray[np.int32],
    balance_groups: bool = False,
    upweight: bool = False,
) -> NDArray[np.float64]:
    """Compute weights for all samples.

    Args:
        train: The training data.
        balance_groups: Whether to balance the groups. When False, the groups are balanced as in
            `Kamiran and Calders 2012 <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_.
            When True, the groups are numerically balanced. (Default: False)
        upweight: If balance_groups is True, whether to upweight the groups, or to downweight
            them. Downweighting is done by multiplying the weights by the inverse of the group size and
            is more numerically stable for small group sizes. (Default: False)
    Returns:
        A dataframe with the instance weights for each sample in the training data.
    """
    num_samples = len(y)
    s_unique, inv_indices_s, counts_s = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    _, inv_indices_y, counts_y = np.unique(y, return_inverse=True, return_counts=True)
    group_ids = (inv_indices_y * len(s_unique) + inv_indices_s).squeeze()
    gi_unique, inv_indices_gi, counts_joint = np.unique(
        group_ids, return_inverse=True, return_counts=True
    )
    if balance_groups:
        group_weights = (
            # Upweight samples according to the cardinality of their intersectional group
            num_samples / counts_joint
            if upweight
            # Downweight samples according to the cardinality of their intersectional group
            # - this approach should be preferred due to being more numerically stable
            # (very small counts can lead to very large weighted loss values when upweighting)
            else 1 - (counts_joint / num_samples)
        )
    else:
        counts_factorized = np.outer(counts_y, counts_s).flatten()
        group_weights = counts_factorized[gi_unique] / (num_samples * counts_joint)

    return group_weights[inv_indices_gi]


@dataclass
class Majority(BaseEstimator, Method):
    """Simply returns the majority label from the train set."""

    random_state: None = None

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> Self:
        """Fit the model by storing the majority class."""
        classes, counts = np.unique(y, return_counts=True)
        self.majority_class_: np.int32 = classes[np.argmax(counts)]
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict the majority class for all samples."""
        return np.full(X.shape[0], self.majority_class_, dtype=np.int32)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}


@dataclass
class Blind(BaseEstimator, Method):
    """A Random classifier.

    This is useful as a baseline method and operates a 'coin flip' to assign a label.
    Returns a random label.
    """

    random_state: int = 0

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> Self:
        """Fit the model by storing the classes."""
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict a random label for all samples."""
        random_state = reproducible_random_state(self.random_state)
        return random_state.choice(self.classes_, size=X.shape[0], replace=True).astype(
            np.int32
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return asdict(self)
