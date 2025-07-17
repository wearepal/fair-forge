from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from fair_forge.utils import reproducible_random_state

from .definitions import GroupPreMethod

__all__ = ["UpsampleStrategy", "Upsampler"]


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
        y: NDArray[np.int32],
        *,
        groups: NDArray[np.int32],
    ) -> Self:
        s_vals: NDArray[np.int32] = np.unique(groups)
        y_vals: NDArray[np.int32] = np.unique(y)

        segments: list[tuple[np.int32, np.int32]] = list(
            itertools.product(s_vals, y_vals)
        )

        data: dict[tuple[np.int32, np.int32], np.int64] = {}
        for s_val, y_val in segments:
            s_y_mask: NDArray[np.bool] = (groups == s_val) & (y == y_val)
            data[(s_val, y_val)] = np.count_nonzero(s_y_mask)

        percentages: dict[tuple[np.int32, np.int32], np.float64] = {}

        vals = list(data.values())

        for key, length in data.items():
            if self.strategy is UpsampleStrategy.NAIVE:
                percentages[key] = (np.max(vals) / length).astype(np.float64)
            else:
                s_val = key[0]
                y_val = key[1]

                y_eq_y = np.count_nonzero(y == y_val)
                s_eq_s = np.count_nonzero(groups == s_val)

                num_samples = len(y)
                num_batch = length

                percentages[key] = (y_eq_y * s_eq_s / (num_batch * num_samples)).astype(
                    np.float64
                )
        self.percentages = percentages
        print(f"Percentages: {self.percentages}")
        return self

    @overload
    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]: ...
    @overload
    def transform(
        self, X: NDArray[np.float32], *, y: NDArray[np.int32], groups: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]: ...
    def transform(
        self,
        X: NDArray[np.float32],
        *,
        y: NDArray[np.int32] | None = None,
        groups: NDArray[np.int32] | None = None,
    ) -> (
        NDArray[np.float32]
        | tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]
    ):
        if y is None or groups is None:
            print(
                "Warning: Upsampler is in test mode, no upsampling will be performed."
            )
            print(f"{y=}, {groups=}")
            return X  # we're in test mode, no upsampling needed
        print("Upsampling...")
        upsampled: dict[
            tuple[np.int32, np.int32],
            tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]],
        ] = {}
        generator = reproducible_random_state(self.random_state)
        for (s_val, y_val), percentage in self.percentages.items():
            mask: NDArray[np.bool] = (groups == s_val) & (y == y_val)
            segment_len = np.count_nonzero(mask)
            required = round(percentage * segment_len)
            indices: NDArray[np.int64] = generator.choice(
                np.arange(segment_len, dtype=np.int64),
                size=required,
                replace=True,
                shuffle=True,
            )
            upsampled[(s_val, y_val)] = (
                X[mask][indices],
                y[mask][indices],
                groups[mask][indices],
            )

        X_concat = np.concatenate(
            [segment[0] for segment in upsampled.values()], axis=0
        )
        y_concat = np.concatenate(
            [segment[1] for segment in upsampled.values()], axis=0
        )
        groups_concat = np.concatenate(
            [segment[2] for segment in upsampled.values()], axis=0
        )
        breakpoint()
        return X_concat, y_concat, groups_concat

    def fit_transform(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
        """Fit the upsampler and transform the data."""
        self.fit(X, y, groups=groups)
        return self.transform(X, y=y, groups=groups)
