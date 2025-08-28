from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Flag, auto
from typing import Literal, Protocol, override

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from ._metrics_common import renyi_correlation

__all__ = [
    "Float",
    "GroupMetric",
    "LabelType",
    "Metric",
    "MetricAgg",
    "RenyiCorrelation",
    "as_group_metric",
    "cv",
    "prob_neg",
    "prob_pos",
    "tnr",
    "tpr",
]

type Float = float | np.float16 | np.float32 | np.float64
"""Union of common float types."""


class Metric(Protocol):
    @property
    def __name__(self) -> str:
        """The name of the metric."""
        ...

    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        sample_weight: NDArray[np.bool] | None = ...,
    ) -> Float: ...


class GroupMetric(Protocol):
    @property
    def __name__(self) -> str:
        """The name of the metric."""
        ...

    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> Float: ...


type LabelType = Literal["group", "y"]
"""A type for specifying which labels to use (class or group labels)."""


@dataclass
class RenyiCorrelation(GroupMetric):
    """Renyi correlation. Measures how dependent two random variables are.

    As defined in this paper: https://link.springer.com/content/pdf/10.1007/BF02024507.pdf ,
    titled "On Measures of Dependence" by Alfréd Rényi.
    """

    base: LabelType = "group"
    """Which label to use as base to compute the correlation against."""

    @property
    def __name__(self) -> str:
        """The name of the metric."""
        return f"renyi_{self.base}"

    @override
    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> float:
        return renyi_correlation(x=y_true if self.base == "y" else groups, y=y_pred)


def prob_pos(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    sample_weight: NDArray[np.bool] | None = None,
) -> np.float64:
    """Probability of positive prediction.

    example:

        >>> import fair_forge as ff
        >>> y_true = np.array([0, 0, 0, 1], dtype=np.int32)
        >>> y_pred = np.array([0, 1, 0, 1], dtype=np.int32)
        >>> ff.metrics.prob_pos(y_true, y_pred)
        np.float64(0.5)
    """
    _, f_pos, _, t_pos, total = _confusion_matrix(
        y_pred=y_pred, y_true=y_true, sample_weight=sample_weight
    )
    return ((t_pos + f_pos) / total).astype(np.float64)


def prob_neg(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    sample_weight: NDArray[np.bool] | None = None,
) -> np.float64:
    """Probability of negative prediction."""
    t_neg, _, f_neg, _, total = _confusion_matrix(
        y_pred=y_pred, y_true=y_true, sample_weight=sample_weight
    )
    return ((t_neg + f_neg) / total).astype(np.float64)


def tpr(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    sample_weight: NDArray[np.bool] | None = None,
) -> np.float64:
    """True Positive Rate (TPR) or Sensitivity."""
    _, _, f_neg, t_pos, _ = _confusion_matrix(
        y_pred=y_pred, y_true=y_true, sample_weight=sample_weight
    )
    return (t_pos / (t_pos + f_neg)).astype(np.float64)


def tnr(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    sample_weight: NDArray[np.bool] | None = None,
) -> np.float64:
    """True Negative Rate (TNR) or Specificity."""
    t_neg, f_pos, _, _, _ = _confusion_matrix(
        y_pred=y_pred, y_true=y_true, sample_weight=sample_weight
    )
    return (t_neg / (t_neg + f_pos)).astype(np.float64)


def _confusion_matrix(
    *,
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    sample_weight: NDArray[np.bool] | None,
) -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64]:
    """Apply sci-kit learn's confusion matrix.

    We assume that the positive class is 1.

    Returns the 4 entries of the confusion matrix, and the total, as a 5-tuple.
    """
    conf_matr: NDArray[np.int64] = confusion_matrix(
        y_true=y_true, y_pred=y_pred, normalize=None, sample_weight=sample_weight
    )

    labels = np.unique(y_true)
    pos_class = np.int32(1)

    if pos_class not in labels:
        raise ValueError("Positive class specified must exist in the true labels.")

    # Find the index of the positive class
    tp_idx = np.nonzero(labels == pos_class)[0].item()

    true_pos = conf_matr[tp_idx, tp_idx]
    false_pos = conf_matr[:, tp_idx].sum() - true_pos
    false_neg = conf_matr[tp_idx, :].sum() - true_pos
    total = conf_matr.sum()
    true_neg = total - true_pos - false_pos - false_neg
    return true_neg, false_pos, false_neg, true_pos, total


@dataclass
class _AggMetricBase(GroupMetric):
    metric: Metric
    agg_name: str
    remove_score_suffix: bool

    @property
    def __name__(self) -> str:
        """The name of the metric."""
        name = self.metric.__name__
        if self.remove_score_suffix and name.endswith("_score"):
            name = name[:-6]
        return f"{name}_{self.agg_name}"

    def _group_scores(
        self,
        *,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        groups: NDArray[np.int32],
        unique_groups: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        return np.array(
            [
                self.metric(y_true[groups == group], y_pred[groups == group])
                for group in unique_groups
            ],
            dtype=np.float64,
        )


@dataclass
class _BinaryAggMetric(_AggMetricBase):
    aggregator: Callable[[np.float64, np.float64], np.float64]

    @override
    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> Float:
        """Compute the metric for the given predictions and actual values."""
        unique_groups = np.unique(groups)
        assert len(unique_groups) == 2, (
            f"Aggregation metric with {self.agg_name} requires exactly two groups for aggregation"
        )
        group_scores = self._group_scores(
            y_true=y_true, y_pred=y_pred, groups=groups, unique_groups=unique_groups
        )
        return self.aggregator(group_scores[0], group_scores[1])


@dataclass
class _MulticlassAggMetric(_AggMetricBase):
    aggregator: Callable[[NDArray[np.float64]], Float]

    @override
    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> Float:
        """Compute the metric for the given predictions and actual values."""
        unique_groups = np.unique(groups)
        group_scores = self._group_scores(
            y_true=y_true, y_pred=y_pred, groups=groups, unique_groups=unique_groups
        )
        return self.aggregator(group_scores)


class MetricAgg(Flag):
    """Aggregation methods for metrics that are computed per group."""

    INDIVIDUAL = auto()
    """Individual per-group results."""
    DIFF = auto()
    """Difference of the per-group results."""
    MAX = auto()
    """Maximum of the per-group results."""
    MIN = auto()
    """Minimum of the per-group results."""
    MIN_MAX = MIN | MAX
    """Equivalent to ``MIN | MAX``."""
    RATIO = auto()
    """Ratio of the per-group results."""
    DIFF_RATIO = INDIVIDUAL | DIFF | RATIO
    """Equivalent to ``INDIVIDUAL | DIFF | RATIO``."""
    ALL = DIFF_RATIO | MIN_MAX
    """All aggregations."""


def as_group_metric(
    base_metrics: Sequence[Metric],
    agg: MetricAgg = MetricAgg.DIFF_RATIO,
    remove_score_suffix: bool = True,
) -> list[GroupMetric]:
    """Turn a sequence of metrics into a list of group metrics."""
    metrics = []
    for metric in base_metrics:
        if MetricAgg.DIFF in agg:
            metrics.append(
                _BinaryAggMetric(
                    metric=metric,
                    agg_name="diff",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: j - i,
                )
            )
        if MetricAgg.RATIO in agg:
            metrics.append(
                _BinaryAggMetric(
                    metric=metric,
                    agg_name="ratio",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: i / j if j != 0 else np.float64(np.nan),
                )
            )
        if MetricAgg.MIN in agg:
            metrics.append(
                _MulticlassAggMetric(
                    metric=metric,
                    agg_name="min",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=np.min,
                )
            )
        if MetricAgg.MAX in agg:
            metrics.append(
                _MulticlassAggMetric(
                    metric=metric,
                    agg_name="max",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=np.max,
                )
            )
        if MetricAgg.INDIVIDUAL in agg:
            metrics.append(
                _BinaryAggMetric(
                    metric=metric,
                    agg_name="0",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: i,
                )
            )
            metrics.append(
                _BinaryAggMetric(
                    metric=metric,
                    agg_name="1",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: j,
                )
            )
    return metrics


def cv(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    groups: NDArray[np.int32],
) -> Float:
    """Calder-Verwer."""
    unique_groups = np.unique(groups)
    assert len(unique_groups) == 2, (
        f"Calder-Verwer requires exactly two groups, got {len(unique_groups)}"
    )
    group_scores = np.array(
        [
            prob_pos(y_true[groups == group], y_pred[groups == group])
            for group in unique_groups
        ],
        dtype=np.float64,
    )
    return 1 - (group_scores[1] - group_scores[0])
