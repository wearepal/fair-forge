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
        match self.label:
            case "y":
                # Ensure the labels are non-negative and contiguous
                uniques, labels = np.unique(y, return_inverse=True)
                num_label_values = len(uniques)
            case "group":
                uniques, labels = np.unique(groups, return_inverse=True)
                num_label_values = len(uniques)
            case "combination":
                y_values, y_contiguous = np.unique(y, return_inverse=True)
                num_classes = len(y_values)
                group_values, groups_contiguous = np.unique(groups, return_inverse=True)
                num_groups = len(group_values)
                labels = y_contiguous * num_groups + groups_contiguous
                num_label_values = num_classes * num_groups

        # Calculate the proportions of each label
        proportions = np.bincount(labels, minlength=num_label_values) / len(labels)
        return np.log(len(proportions)) - entropy(proportions)


type BinaryAggregation = Literal["diff", "ratio"]


@dataclass
class LabelImbalance(DataMetric):
    """Calculate the binary label imbalance as the difference between the proportions of the two classes.

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
        a = y if self.label == "y" else groups
        label_values, labels = np.unique(a, return_inverse=True)
        label_value_0: np.int32
        label_value_1: np.int32
        if len(label_values) > 2:
            # If there are more than two label values, we use the most common and
            # the least common label values instead.
            sorted_indices = np.bincount(labels).argsort()
            max_index: np.int64 = sorted_indices[-1]
            min_index: np.int64 = sorted_indices[0]
            label_value_0 = label_values[min_index]
            label_value_1 = label_values[max_index]
        elif len(label_values) == 2:
            label_value_0, label_value_1 = label_values.tolist()
        else:
            label_value_1 = label_values[0]
            # If there is only one label value, give the other label value as 0 or 1
            # depending on the value of the first label value.
            label_value_0 = np.int32(0) if label_value_1 != 0 else np.int32(1)

        proportion_0 = np.float64(np.count_nonzero(a == label_value_0) / len(a))
        proportion_1 = np.float64(np.count_nonzero(a == label_value_1) / len(a))
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
