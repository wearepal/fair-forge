from dataclasses import dataclass
import itertools
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from fair_forge.methods import GroupMethod, Method
from fair_forge.utils import reproducible_random_state

from .definitions import GroupDatasetModifier

__all__ = ["GroupPipeline", "UpsampleStrategy", "Upsampler"]


@dataclass
class GroupPipeline(BaseEstimator, GroupMethod):
    """A pipeline that applies a group-based data modification method followed by an estimator."""

    group_data_modifier: GroupDatasetModifier
    """A method to modify the dataset based on group information."""
    estimator: Method
    """An estimator to fit the modified dataset."""
    random_state: int | None = None
    """Random state for reproducibility."""

    def __post_init__(self) -> None:
        self.update_random_state()

    def update_random_state(self) -> None:
        if self.random_state is not None:
            self.group_data_modifier.set_params(random_state=self.random_state)
            self.estimator.set_params(random_state=self.random_state)

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self:
        # Fit the group pre-processing method and transform the data
        self.group_data_modifier.fit(X, y=y, groups=groups)
        X_transformed = self.group_data_modifier.transform(X, is_train=True, is_x=True)
        # Transform the labels
        y_transformed = self.group_data_modifier.transform(y, is_train=True)

        # Fit the estimator with the transformed data
        self.estimator.fit(X_transformed, y=y_transformed)
        self.fitted_ = True
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        # Transform the input data using the group pre-processing method
        X_transformed = self.group_data_modifier.transform(X, is_train=False)
        return self.estimator.predict(X_transformed)

    def set_params(self, **params: Any) -> Self:
        ret = super().set_params(**params)
        self.update_random_state()
        return ret


type UpsampleStrategy = Literal["uniform", "naive"]  # , "preferential"]
"""A type for specifying the strategy to use for upsampling."""


@dataclass
class Upsampler(BaseEstimator, GroupDatasetModifier):
    strategy: UpsampleStrategy = "uniform"
    """The strategy to use for upsampling. Options are 'uniform' and 'naive'."""
    random_state: int = 0
    """Random state for reproducibility."""

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self:
        s_vals: NDArray[np.int32] = np.unique(groups)
        y_vals: NDArray[np.int32] = np.unique(y)

        segments: list[tuple[np.int32, np.int32]] = list(
            itertools.product(s_vals, y_vals)
        )

        data: list[tuple[NDArray[np.bool], np.int64, np.int64, np.int64]] = []
        for s_val, y_val in segments:
            s_y_mask: NDArray[np.bool] = (groups == s_val) & (y == y_val)
            y_eq_y = np.count_nonzero(y == y_val)
            s_eq_s = np.count_nonzero(groups == s_val)
            data.append((s_y_mask, np.count_nonzero(s_y_mask), y_eq_y, s_eq_s))

        percentages: list[tuple[NDArray[np.bool], np.float64]] = []

        vals = list([d[1] for d in data])

        for mask, length, y_eq_y, s_eq_s in data:
            if self.strategy == "naive":
                percentages.append((mask, (np.max(vals) / length).astype(np.float64)))
            else:
                num_samples = len(y)
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
        self, X: NDArray[S], *, is_train: bool = False, is_x: bool = True
    ) -> NDArray[S]:
        if not is_train:
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
