from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import polars as pl
import polars.selectors as cs


class Dataset(NamedTuple):
    """A dataset containing features, labels, and optionally groups.

    Attributes:
        X: Features of the dataset.
        y: Labels of the dataset.
        groups: Optional groups for the dataset.
        name: Name of the dataset.
        feature_grouping: Slices indicating groups of features.
    """

    data: NDArray[np.float32]
    target: NDArray[np.int32]
    groups: NDArray[np.int32]
    name: str
    feature_grouping: list[slice]


class AdultGroup(Enum):
    SEX = "Sex"
    RACE = "Race"


def adult_dataset(
    group: AdultGroup,
    *,
    group_in_features: bool = False,
    binarize_nationality: bool = False,
    binarize_race: bool = False,
) -> Dataset:
    """Load the Adult dataset with specified group information.

    Args:
        group: The group to use for the dataset.

    Returns:
        A Dataset object containing the Adult dataset.
    """
    name = f"Adult {group.value}"
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
        case AdultGroup.SEX:
            groups = (
                df.get_column("sex").cat.starts_with("Male").cast(pl.Int32).to_numpy()
            )
            to_drop = "sex"
        case AdultGroup.RACE:
            # `.to_physical()` converts the categorical column to its physical representation,
            # which is UInt32 by default in Polars.
            groups = df.get_column("race").to_physical().cast(pl.Int32).to_numpy()
            to_drop = "race"
    if not group_in_features:
        df = df.drop(to_drop)
        column_grouping_prefixes.remove(to_drop)

    # Convert categorical columns to one-hot encoded features
    df = df.to_dummies(to_dummies, separator=":")

    feature_grouping: list[slice] = []
    columns = df.columns
    for prefix in column_grouping_prefixes:
        # Find the indices of columns that start with the prefix
        indices = [i for i, col in enumerate(columns) if col.startswith(prefix + ":")]
        if not indices:
            raise ValueError(f"No columns found with prefix '{prefix}'.")
        start = min(indices)
        end = max(indices) + 1
        assert all(
            i in indices for i in range(start, end)
        ), f"The columns correponding to prefix '{prefix}' are not contiguous."
        feature_grouping.append(slice(start, end))

    features = df.cast(pl.Float32).to_numpy()
    return Dataset(
        data=features,
        target=y,
        groups=groups,
        name=name,
        feature_grouping=feature_grouping,
    )
