import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLUMN_RENAMING = {
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

COLUMNS_TO_DROP = ["fnlwgt", "education-num", "income"]


def clean_columns(
    df: pd.DataFrame,
    renaming_map: dict = COLUMN_RENAMING,
    columns_to_drop: list = COLUMNS_TO_DROP,
) -> pd.DataFrame:
    """
    Renames a standard set of columns to use snake_case and drops
    a predefined list of columns (fnlwgt, education-num, income).
    """
    columns_to_drop_in_df = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop_in_df:
        df = df.drop(columns=columns_to_drop_in_df)

    df = df.rename(columns=renaming_map)

    return df


def clean_and_binarize_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'income' column and converts it into a boolean field
    (True for '>50K', False for '=<50K').
    """
    income_col = "income"
    high_income_col = "high_income"
    cleaned_income = df[income_col].astype(str).str.strip().str.strip(".")
    df[income_col] = cleaned_income
    df[high_income_col] = cleaned_income == ">50K"

    return df


def replace_question_marks_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces '?' with np.nan across all columns in the dataframe.
    """
    return df.replace("?", np.nan)


def trim_string(value: Any) -> Any:
    """
    Trims leading and trailing whitespace from a single string.
    Returns the original value if it's not a string (e.g., NaN or numbers).
    """
    if isinstance(value, str):
        return value.strip()

    return value


def trim_dataframe_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detects string (object) columns in a Pandas DataFrame
    and strips whitespace from all values.
    """
    string_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in string_columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
        else:
            df[col] = df[col].apply(trim_string)

    return df


def full_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function that runs all cleaning steps in a logical order.
    """
    df = clean_and_binarize_income(df)
    df = clean_columns(df, COLUMN_RENAMING)
    df = trim_dataframe_whitespace(df)
    df = replace_question_marks_with_nan(df)

    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs full cleaning pipeline and saves result in parquet format.
    """
    df = full_clean(df)

    current_file = Path(__file__).resolve()
    src_directory = current_file.parent.parent

    if str(src_directory) not in sys.path:
        sys.path.append(str(src_directory))

    data_dir = src_directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "cleaned_census_income.parquet"

    df.to_parquet(output_path)
    print(
        f"✅ Saved {df.shape[0]} rows, {df.shape[1]} columns to: {data_dir.name}/{output_path.name}"
    )
