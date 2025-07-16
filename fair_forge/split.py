import itertools

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

__all__ = ["basic_split", "proportional_split"]


def basic_split(
    generator: Generator,
    train_percentage: float,
    *,
    target: NDArray[np.int32],
    groups: NDArray[np.int32],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split the dataset into training and testing sets with a basic split."""
    length = len(target)
    train_size = round(length * train_percentage)
    indices = np.arange(length, dtype=np.int64)
    generator.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices, test_indices


def proportional_split(
    generator: Generator,
    train_percentage: float,
    *,
    target: NDArray[np.int32],
    groups: NDArray[np.int32],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Generate the indices of the train and test splits using a proportional sampling scheme."""
    # local random state that won't affect the global state
    s_vals: list[np.int32] = list(np.unique(groups))
    y_vals: list[np.int32] = list(np.unique(target))

    train_indices: list[NDArray[np.int64]] = []
    test_indices: list[NDArray[np.int64]] = []

    # iterate over all combinations of s and y
    for s, y in itertools.product(s_vals, y_vals):
        # find all indices for this group
        idx = np.nonzero((groups == s) & (target == y))[0]

        # shuffle and take subsets
        generator.shuffle(idx)
        split_indices: int = round(len(idx) * train_percentage)
        # append index subsets to the list of train indices
        train_indices.append(idx[:split_indices])
        test_indices.append(idx[split_indices:])

    train_indices_ = np.concatenate(train_indices, axis=0)
    test_indices_ = np.concatenate(test_indices, axis=0)
    del train_indices
    del test_indices

    num_groups = len(s_vals) * len(y_vals)
    expected_train_len = round(len(target) * train_percentage)
    # assert that we (at least approximately) achieved the specified `train_percentage`
    # the maximum error occurs when all the group splits favor train or all favor test
    assert (
        expected_train_len - num_groups
        <= len(train_indices_)
        <= expected_train_len + num_groups
    )

    return train_indices_, test_indices_
