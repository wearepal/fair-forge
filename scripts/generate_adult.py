"""Transparently show how the UCI Adult dataset was generated from the raw download."""

from pathlib import Path

import polars as pl
import polars.selectors as cs


def run_generate_adult() -> None:
    """Generate the UCI Adult dataset from scratch."""
    # Data column names
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    # Load the data
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "raw"
    train = pl.read_csv(
        data_path / "adult.data",
        has_header=False,
        new_columns=columns,
        null_values=["?", " ?", "? "],
    )
    test = pl.read_csv(
        data_path / "adult.test",
        skip_rows=1,
        has_header=False,
        new_columns=columns,
        null_values=["?", " ?", "? "],
    )

    # Concat the data
    all_data = pl.concat([train, test], how="vertical")

    all_data = all_data.with_columns(cs.string().str.strip_chars())

    # Re-infer schema from the first 100 rows
    inferred_schema = pl.read_csv(all_data.head(100).write_csv().encode()).schema
    all_data = all_data.cast(inferred_schema)  # type: ignore

    # Replace full stop in the label of the test set
    all_data = all_data.with_columns(pl.col("salary").str.replace("<=50K.", "<=50K"))
    all_data = all_data.with_columns(pl.col("salary").str.replace(">50K.", ">50K"))

    # Drop NaNs
    all_data = all_data.drop_nulls()

    # Convert string columns to categorical type
    all_data = all_data.with_columns(cs.string().cast(pl.Categorical))

    assert all_data.shape == (45_222, 15)

    # Save the parquet file
    all_data.write_parquet(base_path / "src" / "fair_forge" / "data" / "adult.parquet")


if __name__ == "__main__":
    run_generate_adult()
