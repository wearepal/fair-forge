import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Flag, StrEnum, auto
from typing import Protocol, override

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


class Metric(Protocol):
    @property
    def __name__(self) -> str:
        """The name of the metric."""
        ...

    def __call__(
        self, y_true: NDArray[np.int32], y_pred: NDArray[np.int32]
    ) -> float: ...


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
    ) -> float: ...


class DependencyTarget(StrEnum):
    """The variable that is compared to the predictions in order to check how similar they are."""

    s = auto()
    y = auto()


@dataclass
class RenyiCorrelation(GroupMetric):
    """Renyi correlation. Measures how dependent two random variables are.

    As defined in this paper: https://link.springer.com/content/pdf/10.1007/BF02024507.pdf ,
    titled "On Measures of Dependence" by Alfréd Rényi.
    """

    base: DependencyTarget = DependencyTarget.s

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
        base_values = y_true if self.base is DependencyTarget.y else groups
        return self._corr(base_values.ravel(), y_pred.ravel())

    @staticmethod
    def _corr(x: NDArray[np.int32], y: NDArray[np.int32]) -> float:
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
                joint[i, k] = _count_true((x == x_val) & (y == y_val)) / total

        marginal_rows = np.sum(joint, axis=0, keepdims=True)
        marginal_cols = np.sum(joint, axis=1, keepdims=True)
        q_matrix = joint / np.sqrt(marginal_rows) / np.sqrt(marginal_cols)
        # singular value decomposition of Q
        singulars = np.linalg.svd(q_matrix, compute_uv=False)

        # return second-largest singular value
        return singulars[1]


def _count_true(mask: np.ndarray) -> int:
    """Count the number of elements that are True."""
    return np.count_nonzero(mask).item()


def prob_pos(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    groups: NDArray[np.int32],
) -> float:
    """Probability of positive prediction."""
    _, f_pos, _, t_pos = _confusion_matrix(y_pred, y_true)
    return (t_pos + f_pos) / len(y_pred)


def prob_neg(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    groups: NDArray[np.int32],
) -> float:
    """Probability of negative prediction."""
    t_neg, _, f_neg, _ = _confusion_matrix(y_pred, y_true)
    return (t_neg + f_neg) / len(y_pred)


def _confusion_matrix(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
) -> tuple[int, int, int, int]:
    """Apply sci-kit learn's confusion matrix.

    We assume that the positive class is 1.

    Returns the 4 entries of the confusion matrix as a 4-tuple.
    """
    conf_matr: NDArray[np.int64] = confusion_matrix(
        y_true=y_true, y_pred=y_pred, normalize=None
    )

    labels = np.unique(y_true)
    pos_class = np.int32(1)

    if pos_class not in labels:
        raise ValueError("Positive class specified must exist in the test set")

    # Find the index of the positive class
    tp_idx = np.nonzero(labels == pos_class)[0].item()

    true_pos = conf_matr[tp_idx, tp_idx]
    false_pos = conf_matr[:, tp_idx].sum() - true_pos
    false_neg = conf_matr[tp_idx, :].sum() - true_pos
    true_neg = conf_matr.sum() - true_pos - false_pos - false_neg
    return true_neg, false_pos, false_neg, true_pos


@dataclass
class _PerSensMetricBase(GroupMetric):
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
    ) -> list[float]:
        return [
            self.metric(y_true[groups == group], y_pred[groups == group])
            for group in unique_groups
        ]


@dataclass
class BinaryPerSensMetric(_PerSensMetricBase):
    aggregator: Callable[[float, float], float]

    @override
    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> float:
        """Compute the metric for the given predictions and actual values."""
        unique_groups = np.unique(groups)
        assert (
            len(unique_groups) == 2
        ), f"PerSensMetric with {self.agg_name} requires exactly two groups for aggregation"
        group_scores = self._group_scores(
            y_true=y_true, y_pred=y_pred, groups=groups, unique_groups=unique_groups
        )
        return self.aggregator(group_scores[0], group_scores[1])


@dataclass
class MulticlassPerSensMetric(_PerSensMetricBase):
    aggregator: Callable[[Sequence[float]], float]

    @override
    def __call__(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> float:
        """Compute the metric for the given predictions and actual values."""
        unique_groups = np.unique(groups)
        group_scores = self._group_scores(
            y_true=y_true, y_pred=y_pred, groups=groups, unique_groups=unique_groups
        )
        return self.aggregator(group_scores)


class PerSens(Flag):
    """Aggregation methods for metrics that are computed per sensitive attributes."""

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
    DIFF_RATIO = DIFF | RATIO
    """Equivalent to ``DIFFS | RATIOS``."""
    ALL = DIFF_RATIO | MIN_MAX
    """All aggregations."""


def per_sens_metrics(
    base_metrics: Sequence[Metric],
    per_sens: PerSens,
    remove_score_suffix: bool = True,
) -> list[GroupMetric]:
    """Create per-sensitive attribute metrics from base metrics."""
    metrics = []
    for metric in base_metrics:
        if per_sens & PerSens.DIFF:
            metrics.append(
                BinaryPerSensMetric(
                    metric=metric,
                    agg_name="diff",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: i - j,
                )
            )
        if per_sens & PerSens.RATIO:
            metrics.append(
                BinaryPerSensMetric(
                    metric=metric,
                    agg_name="ratio",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=lambda i, j: i / j if j != 0 else math.nan,
                )
            )
        if per_sens & PerSens.MIN:
            metrics.append(
                MulticlassPerSensMetric(
                    metric=metric,
                    agg_name="min",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=min,
                )
            )
        if per_sens & PerSens.MAX:
            metrics.append(
                MulticlassPerSensMetric(
                    metric=metric,
                    agg_name="max",
                    remove_score_suffix=remove_score_suffix,
                    aggregator=max,
                )
            )
    return metrics
