from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from enum import Enum, Flag, auto
from hashlib import sha1
from typing import ClassVar, Protocol, override

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

__all__ = [
    "DependencyTarget",
    "Float",
    "GroupMetric",
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


class DependencyTarget(Enum):
    """The variable that is compared to the predictions in order to check how similar they are."""

    S = "s"
    Y = "y"


@dataclass
class RenyiCorrelation(GroupMetric):
    """Renyi correlation. Measures how dependent two random variables are.

    As defined in this paper: https://link.springer.com/content/pdf/10.1007/BF02024507.pdf ,
    titled "On Measures of Dependence" by Alfréd Rényi.
    """

    base: DependencyTarget = DependencyTarget.S

    @property
    def __name__(self) -> str:
        """The name of the metric."""
        return f"renyi_{self.base.value}"

    @override
    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> float:
        base_values = y_true if self.base is DependencyTarget.Y else groups
        x: NDArray[np.int32] = base_values.ravel()
        y: NDArray[np.int32] = y_pred.ravel()
        x_vals = np.unique(x)
        y_vals = np.unique(y)
        if len(x_vals) < 2 or len(y_vals) < 2:
            return 1.0

        total = len(x)
        assert total == len(y)

        joint = np.empty((len(x_vals), len(y_vals)))

        for i, x_val in enumerate(x_vals):
            for k, y_val in enumerate(y_vals):
                # count how often x_val and y_val co-occur
                joint[i, k] = (
                    np.count_nonzero((x == x_val) & (y == y_val)).item() / total
                )

        marginal_rows = np.sum(joint, axis=0, keepdims=True)
        marginal_cols = np.sum(joint, axis=1, keepdims=True)
        q_matrix = joint / np.sqrt(marginal_rows) / np.sqrt(marginal_cols)
        # singular value decomposition of Q
        singulars = np.linalg.svd(q_matrix, compute_uv=False)

        # return second-largest singular value
        return singulars[1]


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


class _HashableArray:
    """A hashable wrapper for numpy arrays."""

    def __init__(
        self, array: NDArray[np.int32], *, unsafe_no_copy: bool = False
    ) -> None:
        if not unsafe_no_copy:
            # We copy the array because numpy arrays are mutable
            # and we cannot control their mutability outside this class.
            self.__array = np.copy(array, order="C")
        else:
            self.__array = array

    def __hash__(self) -> int:
        return int(sha1(self.__array.view(np.uint8)).hexdigest(), 16)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashableArray):
            return np.array_equal(self.__array, other.__array)
        return False

    def unwrap(self) -> NDArray[np.int32]:
        """Return the original numpy array.

        This method returns a copy of the original array to ensure that
        the original data is not modified outside of this class.
        """
        return self.__array.copy()


@dataclass
class _AggMetricBase(GroupMetric):
    metric: Metric
    agg_name: str
    remove_score_suffix: bool
    score_cache: ClassVar[
        dict[
            tuple[Metric, _HashableArray, _HashableArray, _HashableArray],
            NDArray[np.float64],
        ]
    ] = {}

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
        metric_is_hashable = isinstance(self.metric, Hashable)  # type: ignore
        if metric_is_hashable:
            # Check if the scores are already cached
            # For the lookup, we use an unsafe no-copy version of the wrapper
            lookup_key = (
                self.metric,
                _HashableArray(y_true, unsafe_no_copy=True),
                _HashableArray(y_pred, unsafe_no_copy=True),
                _HashableArray(groups, unsafe_no_copy=True),
            )
            if (group_scores := self.score_cache.get(lookup_key)) is not None:
                # `group_scores` is a very small array, so we can safely copy it
                return group_scores.copy()

        # Compute the group scores
        group_scores = np.array(
            [
                self.metric(y_true[groups == group], y_pred[groups == group])
                for group in unique_groups
            ],
            dtype=np.float64,
        )

        if metric_is_hashable:
            key = (
                self.metric,
                _HashableArray(y_true),
                _HashableArray(y_pred),
                _HashableArray(groups),
            )
            # `group_scores` is a very small array, so we can safely copy it
            self.score_cache[key] = group_scores.copy()
        return group_scores


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
        if agg & MetricAgg.DIFF:
            metrics.append(
                _BinaryAggMetric(
                    metric=metric,
                    agg_name="diff",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: j - i,
                )
            )
        if agg & MetricAgg.RATIO:
            metrics.append(
                _BinaryAggMetric(
                    metric=metric,
                    agg_name="ratio",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: i / j if j != 0 else np.float64(np.nan),
                )
            )
        if agg & MetricAgg.MIN:
            metrics.append(
                _MulticlassAggMetric(
                    metric=metric,
                    agg_name="min",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=np.min,
                )
            )
        if agg & MetricAgg.MAX:
            metrics.append(
                _MulticlassAggMetric(
                    metric=metric,
                    agg_name="max",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=np.max,
                )
            )
        if agg & MetricAgg.INDIVIDUAL:
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
