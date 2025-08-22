from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray
import polars as pl
import polars.selectors as cs

from fair_forge.utils import reproducible_random_state

__all__ = [
    "AdultGroup",
    "Dataset",
    "GroupDataset",
    "grouping_by_prefix",
    "load_adult",
    "load_dummy_dataset",
    "load_ethicml_toy",
]


class Dataset(Protocol):
    data: NDArray
    target: NDArray
    feature_names: list[str]


class GroupDataset(NamedTuple):
    """A dataset containing features, labels, and groups.

    Args:
        data: Features of the dataset.
        target: Labels of the dataset.
        groups: Groups of the dataset.
        name: Name of the dataset.
        feature_grouping: Slices indicating groups of features.
        feature_names: Names of the features in the dataset.
    """

    data: NDArray[np.float32]
    target: NDArray[np.int32]
    groups: NDArray[np.int32]
    name: str
    feature_grouping: list[slice]
    feature_names: list[str]


type AdultGroup = Literal["Sex", "Race"]


def load_adult(
    group: AdultGroup,
    *,
    group_in_features: bool = False,
    binarize_nationality: bool = False,
    binarize_race: bool = False,
) -> GroupDataset:
    """Load the Adult dataset with specified group information.

    Args:
        group: The group to use for the dataset.

    Returns:
        A Dataset object containing the Adult dataset.
    """
    name = f"Adult {group}"
    if binarize_nationality:
        name += ", binary nationality"
    if binarize_race:
        name += ", binary race"
    if group_in_features:
        name += ", group in features"
    base_path = Path(__file__).parent
    df = pl.read_parquet(base_path / "data" / "adult.parquet")

    y = df.get_column("salary").cat.starts_with(">50K").cast(pl.Int32).to_numpy()
    df = df.drop("salary")

    df = df.drop("fnlwgt")

    column_grouping_prefixes = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    to_dummies = cs.categorical()

    if binarize_race:
        df = df.with_columns(
            pl.col("race").replace_strict(
                {"White": "White"},
                default="Other",
                return_dtype=pl.Enum(["White", "Other"]),
            )
        )
        to_dummies = to_dummies | cs.by_name("race")
    if binarize_nationality:
        df = df.with_columns(
            pl.col("native-country").replace_strict(
                {"United-States": "United-States"},
                default="Other",
                return_dtype=pl.Enum(["United-States", "Other"]),
            )
        )
        to_dummies = to_dummies | cs.by_name("native-country")

    groups: NDArray[np.int32]
    to_drop: str
    match group:
        case "Sex":
            groups = (
                df.get_column("sex").cat.starts_with("Male").cast(pl.Int32).to_numpy()
            )
            to_drop = "sex"
        case "Race":
            # `.to_physical()` converts the categorical column to its physical representation,
            # which is UInt32 by default in Polars.
            groups = df.get_column("race").to_physical().cast(pl.Int32).to_numpy()
            to_drop = "race"
        case _:
            raise ValueError(f"Invalid group: {group}")
    if not group_in_features:
        df = df.drop(to_drop)
        column_grouping_prefixes.remove(to_drop)

    # Convert categorical columns to one-hot encoded features
    df = df.to_dummies(to_dummies, separator=":")

    columns = df.columns
    feature_grouping = grouping_by_prefix(
        columns=columns, prefixes=[f"{col}:" for col in column_grouping_prefixes]
    )

    features = df.cast(pl.Float32).to_numpy()
    return GroupDataset(
        data=features,
        target=y,
        groups=groups,
        name=name,
        feature_grouping=feature_grouping,
        feature_names=columns,
    )


def grouping_by_prefix(*, columns: list[str], prefixes: list[str]) -> list[slice]:
    """Create slices for feature grouping based on column prefixes."""
    feature_grouping: list[slice] = []
    for prefix in prefixes:
        # Find the indices of columns that start with the prefix
        indices = [i for i, col in enumerate(columns) if col.startswith(prefix)]
        if not indices:
            raise ValueError(f"No columns found with prefix '{prefix}'.")
        start = min(indices)
        end = max(indices) + 1
        assert all(i in indices for i in range(start, end)), (
            f"The columns correponding to prefix '{prefix}' are not contiguous."
        )
        feature_grouping.append(slice(start, end))
    return feature_grouping


def load_dummy_dataset(seed: int) -> GroupDataset:
    """Load a dummy dataset for testing purposes, based on a mixture of 2 2D Gaussians.

    The groups are random.

    Args:
        seed: Random seed for reproducibility.
    """
    generator = reproducible_random_state(seed)

    n_samples = 100
    n_features = 2
    n_groups = 2

    # Diagonal covariance matrix for the 2D Gaussian
    cov = np.eye(n_features)  # Identity matrix for covariance
    # First generate samples for the first class (n=n_samples // 2)
    x1 = generator.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_samples // 2)
    y1 = np.zeros(n_samples // 2, dtype=np.int32)
    groups1 = generator.integers(0, n_groups, size=n_samples // 2, dtype=np.int32)
    # Then generate samples for the second class (n=n_samples // 2)
    x2 = generator.multivariate_normal(mean=[1.5, 1.5], cov=cov, size=n_samples // 2)
    y2 = np.ones(n_samples // 2, dtype=np.int32)
    groups2 = generator.integers(0, n_groups, size=n_samples // 2, dtype=np.int32)
    # Concatenate the samples
    x = np.concatenate((x1, x2), axis=0).astype(np.float32)
    y = np.concatenate((y1, y2), axis=0)
    groups = np.concatenate((groups1, groups2), axis=0)
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    # Create feature grouping (no groupings)
    feature_grouping = []
    name = "Dummy Dataset"
    return GroupDataset(
        data=x,
        target=y,
        groups=groups,
        name=name,
        feature_grouping=feature_grouping,
        feature_names=feature_names,
    )


def load_ethicml_toy(group_in_features: bool = False) -> GroupDataset:
    """Load the EthicML toy dataset."""
    base_path = Path(__file__).parent
    df = pl.read_parquet(base_path / "data" / "toy.parquet")
    y = df.get_column("decision").cast(pl.Int32).to_numpy()
    df = df.drop("decision")
    groups = df.get_column("sensitive-attr").cast(pl.Int32).to_numpy()
    if not group_in_features:
        # If the group is not supposed to be in the features, we drop it
        df = df.drop("sensitive-attr")
    discrete_columns = ["disc_1", "disc_2"]
    df = df.to_dummies(discrete_columns, separator=":")
    features = df.cast(pl.Float32).to_numpy()
    feature_names = df.columns
    feature_grouping = grouping_by_prefix(
        columns=feature_names, prefixes=discrete_columns
    )
    return GroupDataset(
        data=features,
        target=y,
        groups=groups,
        name="Toy",
        feature_grouping=feature_grouping,
        feature_names=feature_names,
    )
