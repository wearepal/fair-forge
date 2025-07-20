from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from fair_forge.methods import Method
from fair_forge.utils import reproducible_random_state

from .definitions import GroupPreMethod

__all__ = ["EstimatorForTransformedLabels", "UpsampleStrategy", "Upsampler"]


@dataclass
class EstimatorForTransformedLabels(BaseEstimator):
    estimator: Method

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        *,
        targets: NDArray[np.int32],
    ) -> Self:
        self.estimator.fit(X, targets)
        self.fitted_ = True
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        return self.estimator.predict(X)


class UpsampleStrategy(Enum):
    """Strategy for upsampling."""

    UNIFORM = "uniform"
    # PREFERENTIAL = "preferential"
    NAIVE = "naive"


@dataclass
class Upsampler(BaseEstimator, GroupPreMethod):
    strategy: UpsampleStrategy = UpsampleStrategy.UNIFORM
    random_state: int = 0

    def fit(
        self,
        X: NDArray[np.float32],
        *,
        targets: NDArray[np.int32],
        groups: NDArray[np.int32],
    ) -> Self:
        s_vals: NDArray[np.int32] = np.unique(groups)
        y_vals: NDArray[np.int32] = np.unique(targets)

        segments: list[tuple[np.int32, np.int32]] = list(
            itertools.product(s_vals, y_vals)
        )

        data: list[tuple[NDArray[np.bool], np.int64, np.int64, np.int64]] = []
        for s_val, y_val in segments:
            s_y_mask: NDArray[np.bool] = (groups == s_val) & (targets == y_val)
            y_eq_y = np.count_nonzero(targets == y_val)
            s_eq_s = np.count_nonzero(groups == s_val)
            data.append((s_y_mask, np.count_nonzero(s_y_mask), y_eq_y, s_eq_s))

        percentages: list[tuple[NDArray[np.bool], np.float64]] = []

        vals = list([d[1] for d in data])

        for mask, length, y_eq_y, s_eq_s in data:
            if self.strategy is UpsampleStrategy.NAIVE:
                percentages.append((mask, (np.max(vals) / length).astype(np.float64)))
            else:
                num_samples = len(targets)
                num_batch = length

                percentages.append(
                    (
                        mask,
                        (y_eq_y * s_eq_s / (num_batch * num_samples)).astype(
                            np.float64
                        ),
                    )
                )
        self.percentages_ = percentages
        return self

    def transform[S: np.generic](
        self,
        X: NDArray[S],
        *,
        targets: NDArray[np.int32] | None = None,
        groups: NDArray[np.int32] | None = None,
    ) -> NDArray[S]:
        if targets is None or groups is None:
            return X  # we're in test mode, no upsampling needed
        upsampled: list[NDArray[S]] = []
        generator = reproducible_random_state(self.random_state)
        for mask, percentage in self.percentages_:
            segment_len = np.count_nonzero(mask)
            required = round(percentage * segment_len)
            indices: NDArray[np.int64] = generator.choice(
                np.arange(segment_len, dtype=np.int64),
                size=required,
                replace=True,
                shuffle=True,
            )
            upsampled.append(X[mask][indices])

        return np.concatenate(upsampled, axis=0)

    def fit_transform(
        self,
        X: NDArray[np.float32],
        y: None,
        *,
        targets: NDArray[np.int32],
        groups: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """Fit the upsampler and transform the data."""
        self.fit(X, targets=targets, groups=groups)
        return self.transform(X, targets=targets, groups=groups)
