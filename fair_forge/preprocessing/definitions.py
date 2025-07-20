from typing import Protocol, Self, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.metadata_routing import MetadataRequest

__all__ = ["GroupPreMethod", "Preprocessor"]


class _PreprocessorBase(Protocol):
    """A protocol for preprocessing methods."""

    def get_params(self, deep: bool = ...) -> dict[str, object]: ...
    def get_metadata_routing(self) -> MetadataRequest: ...


class Preprocessor(_PreprocessorBase, Protocol):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> Self:
        """Fit the preprocessor to the data."""
        ...

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform the data using the fitted preprocessor."""
        ...


class GroupPreMethod(_PreprocessorBase, Protocol):
    def fit(
        self,
        X: NDArray[np.float32],
        *,
        targets: NDArray[np.int32],
        groups: NDArray[np.int32],
    ) -> Self:
        """Fit the preprocessing method to the data with group information."""
        ...

    def transform[S: np.generic](
        self,
        X: NDArray[S],
        *,
        targets: NDArray[np.int32] | None = None,
        groups: NDArray[np.int32] | None = None,
    ) -> NDArray[S]: ...

    def fit_transform(
        self,
        X: NDArray[np.float32],
        y: None,
        *,
        targets: NDArray[np.int32],
        groups: NDArray[np.int32],
    ) -> NDArray[np.float32]: ...
