"""
Simple benchmark comparing Pandas vs Polars for loading and cleaning census data.

Result:
While Polars is significantly faster than Pandas relativley speaking (2x+ speedup),
the absolute time difference is small (fractions of a second) for this dataset size.
Although Pandas has been around for much longer, and therfore is more stable and feature-rich,
Polars has been catching up. For example skit-learn now has support for Polars DataFrames.
However, there is still room to improve, e.g. pandas is still the best choice with seaborn.
As a result, I selected Polars for the bulk of this project, and pandas for edge cases.
"""

import time
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl

from income_predict_d100_d400.robust_paths import DATA_DIR

PARQUET_PATH = DATA_DIR / "census_income.parquet"

COLUMN_RENAMING: Dict[str, str] = {
    "age": "age",
    "workclass": "work_class",
    "education": "education",
    "marital-status": "marital_status",
    "occupation": "occupation",
    "relationship": "relationship",
    "race": "race",
    "sex": "sex",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "hours-per-week": "hours_per_week",
    "native-country": "native_country",
    "income": "income",
}

COLUMNS_TO_DROP: list[str] = ["fnlwgt", "education-num", "income"]

EDUCATION_ORDER: Dict[str, int] = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


def pandas_load_and_clean() -> pd.DataFrame:
    """Load and clean data using pandas."""
    df = pd.read_parquet(PARQUET_PATH)
    df.insert(0, "unique_id", range(len(df)))
    df["income"] = df["income"].astype(str).str.strip().str.strip(".")
    df["high_income"] = df["income"] == ">50K"
    df = df.rename(columns=COLUMN_RENAMING)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    df = df.replace("?", np.nan)
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    df["education"] = df["education"].map(EDUCATION_ORDER)
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]
    df = df.drop(columns=["capital_gain", "capital_loss"])
    df["is_female"] = df["sex"] == "Female"
    df = df.drop(columns=["sex"])
    df["is_white"] = df["race"] == "White"
    df["is_black"] = df["race"] == "Black"
    df = df.drop(columns=["race"])

    # New logic for is_married_healthy
    df["is_married_healthy"] = df["marital_status"].isin(
        ["Married-civ-spouse", "Married-AF-spouse"]
    )
    df = df.drop(columns=["marital_status"])

    return df


def polars_load_and_clean() -> pl.DataFrame:
    """Load and clean data using polars."""
    df = pl.read_parquet(PARQUET_PATH)
    df = df.with_row_index("unique_id")
    df = df.with_columns(
        [
            pl.col("income").str.strip_chars().str.strip_chars(".").alias("income"),
            (pl.col("income").str.strip_chars().str.strip_chars(".") == ">50K").alias(
                "high_income"
            ),
        ]
    )
    df = df.rename({k: v for k, v in COLUMN_RENAMING.items() if k in df.columns})
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in string_cols])
    df = df.with_columns(
        [
            pl.when(pl.col(c) == "?").then(None).otherwise(pl.col(c)).alias(c)
            for c in string_cols
            if c in df.columns
        ]
    )
    df = df.drop([c for c in COLUMNS_TO_DROP if c in df.columns])
    df = df.with_columns(pl.col("education").replace_strict(EDUCATION_ORDER))
    df = df.with_columns(
        (pl.col("capital_gain") - pl.col("capital_loss")).alias("capital_net")
    )
    df = df.drop(["capital_gain", "capital_loss"])
    df = df.with_columns((pl.col("sex") == "Female").alias("is_female")).drop("sex")
    df = df.with_columns(
        [
            (pl.col("race") == "White").alias("is_white"),
            (pl.col("race") == "Black").alias("is_black"),
        ]
    ).drop("race")

    df = df.with_columns(
        pl.col("marital_status")
        .is_in(["Married-civ-spouse", "Married-AF-spouse"])
        .alias("is_married_healthy")
    ).drop("marital_status")

    return df


def main() -> None:
    """Run the benchmark comparison."""
    print(f"Data file: {PARQUET_PATH}")
    print("-" * 50)

    start = time.perf_counter()
    pandas_df = pandas_load_and_clean()
    pandas_time = time.perf_counter() - start
    print(f"Pandas:  {pandas_time:.4f}s  ({len(pandas_df)} rows)")

    start = time.perf_counter()
    polars_df = polars_load_and_clean()
    polars_time = time.perf_counter() - start
    print(f"Polars:  {polars_time:.4f}s  ({len(polars_df)} rows)")

    print("-" * 50)
    speedup = pandas_time / polars_time
    winner = "Polars" if speedup > 1 else "Pandas"
    print(f"Speedup: {speedup:.2f}x ({winner} is faster)")


if __name__ == "__main__":
    main()
