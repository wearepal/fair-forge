from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
import polars as pl
from scipy.stats import entropy

from ._metrics_common import renyi_correlation
from .datasets import GroupDataset
from .metrics import Float, LabelType

__all__ = [
    "DataMetric",
    "DistanceFromUniform",
    "LabelImbalance",
    "data_overview",
    "hgr_corr",
]


class DataMetric(Protocol):
    """Protocol for data metrics that can be calculated based on labels and groups."""

    @property
    def __name__(self) -> str:
        """The name of the metric."""
        ...

    def __call__(self, y: NDArray[np.int32], groups: NDArray[np.int32]) -> Float:
        """Calculate the metric based on the provided labels and groups."""
        ...


@dataclass
class DistanceFromUniform(DataMetric):
    """Kullback-Leibler distance of the distribution of a label from uniformity.

    It is assumed that all possible label values are present in the data.

    Example:

        >>> y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        >>> groups = np.zeros_like(y, dtype=np.int32)
        >>> dist = DistanceFromUniform("y")
        >>> dist(y, groups)
        np.float64(0.0)
        >>> y = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
        >>> round(dist(y, groups), 3)
        np.float64(0.057)
        >>> dist = DistanceFromUniform("combination")
        >>> y = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.int32)
        >>> groups = np.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=np.int32)
        >>> dist(y, groups)
        np.float64(0.0)
    """

    label: LabelType | Literal["combination"]

    @property
    def __name__(self) -> str:
        return f"{self.label}_distance_from_uniform"

    def __call__(self, y: NDArray[np.int32], groups: NDArray[np.int32]) -> np.float64:
        proportions, _ = _proportions(self.label, y, groups)
        return np.log(len(proportions)) - entropy(proportions)


def _proportions(
    label: LabelType | Literal["combination"],
    y: NDArray[np.int32],
    groups: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Calculate the proportions of each label value.

    This function returns two arrays:

    The first array contains the proportions of each label value in the data. The array
    is sorted by the label values, i.e., the first element corresponds to the smallest
    label value, the second element to the second smallest label value, etc.

    The second array contains the original label values, sorted in the same way the
    first array is sorted.

    Example:

        >>> y = np.array([2, -1, 2, 2, -1, -1, 2, 2], dtype=np.int32)
        >>> groups = np.zeros_like(y, dtype=np.int32)
        >>> _proportions("y", y, groups)
        (array([0.375, 0.625]), array([-1,  2], dtype=int32))
    """
    contiguous_labels: NDArray[np.int64]
    index_to_label_map: NDArray[np.int32]
    match label:
        case "y":
            # Ensure the labels are non-negative and contiguous
            index_to_label_map, contiguous_labels = np.unique(y, return_inverse=True)
        case "group":
            index_to_label_map, contiguous_labels = np.unique(
                groups, return_inverse=True
            )
        case "combination":
            y_values, y_contiguous = np.unique(y, return_inverse=True)
            num_classes = len(y_values)
            group_values, groups_contiguous = np.unique(groups, return_inverse=True)
            num_groups = len(group_values)
            contiguous_labels = y_contiguous * num_groups + groups_contiguous
            num_label_values = num_classes * num_groups
            # TODO: find a better way to create a label value mapping here
            index_to_label_map = np.arange(num_label_values, dtype=np.int32)

    # Calculate the proportions of each label
    return (
        np.bincount(contiguous_labels, minlength=len(index_to_label_map))
        / len(contiguous_labels),
        index_to_label_map,
    )


type BinaryAggregation = Literal["diff", "ratio"]


@dataclass
class LabelImbalance(DataMetric):
    """Calculate the imbalance in the class labels or the group labels.

    The imbalance can be expressed as either a difference or a ratio. In the case of
    the difference, positive values indicate an imbalance favoring the positive
    label, while negative values indicate an imbalance favoring the negative label.
    The range of the imbalance is [-1, 1], where 0 indicates a perfect balance
    between the two classes.

    In the case of the ratio, the values range from 0 to infinity, where 1 indicates a
    perfect balance between the two classes. If only one label not present, the
    result is positive infinity. In the case of binary labels, the ratio is greater
    than 1 if the positive label is more common than the negative label, and less than
    1 if the negative label is more common than the positive label.

    If there are more than two label values, the most common and least common label
    values are used to calculate the imbalance. In this case, the result is always
    non-negative for the difference and always greater than or equal to 1 for the ratio.

    Example:

        >>> y = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        >>> groups = np.zeros_like(y, dtype=np.int32)
        >>> binary_class_imbalance = LabelImbalance("y")
        >>> round(binary_class_imbalance(y, groups), 2)
        np.float64(1.5)
        >>> y = np.array([0, 0, 0, 0, 1], dtype=np.int32)
        >>> round(binary_class_imbalance(y, groups), 2)
        np.float64(0.25)
        >>> # Example with only one type of label
        >>> y = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        >>> round(binary_class_imbalance(y, groups), 2)
        np.float64(inf)
        >>> # Example with non-binary labels
        >>> y = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
        >>> round(binary_class_imbalance(y, groups), 2)
        np.float64(3.0)
    """

    label: LabelType
    agg: BinaryAggregation = "ratio"

    @property
    def __name__(self) -> str:
        return f"{self.label}_imbalance_{self.agg}"

    def __call__(self, y: NDArray[np.int32], groups: NDArray[np.int32]) -> np.float64:
        proportions, _ = _proportions(self.label, y, groups)
        if len(proportions) > 2:
            # If there are more than two label values, we use the most common and
            # the least common label values instead.
            sorted_proportions = np.sort(proportions)
            proportion_0 = sorted_proportions[0]
            proportion_1 = sorted_proportions[-1]
        elif len(proportions) == 2:
            # The way the `proportions` array is sorted, the first element corresponds
            # to the smallest label value. We assume that the smallest label value is
            # the negative label and the largest label value is the positive.
            proportion_0, proportion_1 = proportions[0], proportions[1]
        else:
            proportion_1 = proportions[0]
            # If there is only one label value, we pretend that the other label values
            # have a proportion of 0.
            proportion_0 = np.float64(0)
        match self.agg:
            case "diff":
                return proportion_1 - proportion_0
            case "ratio":
                if proportion_0 == 0:
                    inf = np.float64(np.inf)
                    return inf if proportion_1 > 0 else -inf
                return proportion_1 / proportion_0


def hgr_corr(y: NDArray[np.int32], groups: NDArray[np.int32]) -> np.float64:
    """Calculate the Hirschfeld-Gebelein-Rényi correlation between y labels and groups.

    The result ranges from 0 to 1, where 0 indicates no correlation and 1 indicates
    perfect correlation. Note that the way "correlation" is understood here, anti-
    correlation also counts as correlation. The metric measures the highest possible
    correlation that can be achieved by transforming the values in any possible way.

    Example:

        >>> y = np.array([1, 0, 1, 0, 1, 1], dtype=np.int32)
        >>> groups = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)
        >>> round(hgr_corr(y, groups), 3)
        np.float64(0.707)
        >>> # Example with perfect anti-correlation
        >>> y = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
        >>> groups = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        >>> round(hgr_corr(y, groups), 3)
        np.float64(1.0)
        >>> # Example with random classes and groups
        >>> gen = np.random.Generator(np.random.MT19937(42))
        >>> y = gen.integers(0, 2, size=1000, dtype=np.int32)
        >>> groups = gen.integers(0, 2, size=1000, dtype=np.int32)
        >>> round(hgr_corr(y, groups), 3)
        np.float64(0.004)
    """
    return renyi_correlation(y, groups)


def data_overview(ds: GroupDataset) -> pl.DataFrame:
    data_metrics: list[DataMetric] = [
        DistanceFromUniform("y"),
        DistanceFromUniform("group"),
        DistanceFromUniform("combination"),
        LabelImbalance("y", "diff"),
        LabelImbalance("group", "diff"),
        LabelImbalance("y", "ratio"),
        LabelImbalance("group", "ratio"),
        hgr_corr,
    ]
    y = ds.target
    groups = ds.groups
    return pl.DataFrame({metric.__name__: metric(y, groups) for metric in data_metrics})
