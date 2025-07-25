"""Utility functions for neural networks in JAX."""

from collections.abc import Generator

import jax
from jax import Array

from fair_forge.utils import batched

__all__ = ["grad_reverse", "iterate_forever"]


@jax.custom_vjp
def grad_reverse(x: Array, lambda_: float) -> Array:
    """Gradient reversal layer for JAX."""
    return x


def _grad_reverse_fwd(x: Array, lambda_: float) -> tuple[Array, tuple[float]]:
    # Forward pass: just return x and save lambda_ for backward
    return x, (lambda_,)


def _grad_reverse_bwd(res: tuple[float], g: Array) -> tuple[Array, None]:
    # Backward pass: reverse and scale the gradient
    (lambda_,) = res
    return (-g * lambda_, None)  # None for lambda_ grad


# Register the custom VJP
grad_reverse.defvjp(_grad_reverse_fwd, _grad_reverse_bwd)


def iterate_forever[T: Array, *S](
    data: tuple[T, *S],
    *,
    batch_size: int,
    seed: int = 0,
) -> Generator[tuple[T, *S], None, None]:
    """Yield batches of the data tuple forever.

    Use `itertools.islice` to limit the number of batches.
    """
    elem = data[0]
    assert all(d.shape[0] == elem.shape[0] for d in data), (  # type: ignore
        "All elements of data must have the same first dimension."
    )
    assert batch_size > 0, "Batch size must be greater than 0."
    assert batch_size <= elem.shape[0], (
        "Batch size must be less than or equal to the number of samples."
    )
    len_data = elem.shape[0]
    key = jax.random.key(seed)
    while True:
        # First generate shuffled indices.
        key, subkey = jax.random.split(key)
        shuffled_indices = jax.random.permutation(subkey, len_data)

        # Then yield the data in batches.
        for slice_ in batched(len_data, batch_size, drop_last=True):
            batch_indices = shuffled_indices[slice_]
            yield tuple(d[batch_indices] for d in data)  # type: ignore
