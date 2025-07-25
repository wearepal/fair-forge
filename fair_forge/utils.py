"""Utility functions for Fair Forge."""

from collections.abc import Generator

import numpy as np

__all__ = ["batched", "reproducible_random_state"]


def reproducible_random_state(seed: int) -> np.random.Generator:
    """Create a random state that is reproducible across Python versions and platforms."""
    # MT19937 isn't the best random number generator, but it's reproducible, so we're using it.
    return np.random.Generator(np.random.MT19937(seed))


def batched(
    len_data: int, batch_size: int, *, drop_last: bool = False
) -> Generator[slice, None, None]:
    """Yield slices of indices for batching data.

    Args:
        len_data: The total number of data points.
        batch_size: The size of each batch.
        drop_last: If True, the last batch will be dropped if it is smaller than batch_size.
    """
    for start in range(0, len_data, batch_size):
        end = start + batch_size
        if end > len_data:
            if drop_last:
                # If the last batch is smaller than batch_size, we skip it.
                break
            else:
                end = len_data
        yield slice(start, end)
