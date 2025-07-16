from typing import Protocol

from numpy.typing import NDArray


class Preprocessor(Protocol):
    """A protocol for preprocessing methods."""

    def fit(self, X: NDArray) -> "Preprocessor":
        """Fit the preprocessor to the data."""
        ...

    def transform(self, X: NDArray) -> NDArray:
        """Transform the data using the fitted preprocessor."""
        ...
