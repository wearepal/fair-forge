from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.metadata_routing import MetadataRequest

__all__ = ["GroupDatasetModifier", "Preprocessor"]


class _PreprocessorBase(Protocol):
    """A protocol for preprocessing methods."""

    def get_params(self, deep: bool = ...) -> dict[str, object]: ...
    def set_params(self, **kwargs: Any) -> Self: ...
    def get_metadata_routing(self) -> MetadataRequest: ...


class Preprocessor(_PreprocessorBase, Protocol):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> Self:
        """Fit the preprocessor to the data."""
        ...

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform the data using the fitted preprocessor."""
        ...


class GroupDatasetModifier(_PreprocessorBase, Protocol):
    """A transformation which modifies both the dataset and the labels based on group information."""

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self:
        """Fit the preprocessing method to the data with group information."""
        ...

    def transform[S: np.generic](
        self, X: NDArray[S], *, is_train: bool = False, is_x: bool = False
    ) -> NDArray[S]: ...
