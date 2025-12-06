from typing import Any

import numpy as np
import pandas as pd

COLUMN_RENAMING = {
    "age": "age",
    "workclass": "work_class",
    "fnlwgt": "final_weight",  # census demographic weight
    "education": "education",
    "education-num": "education_years",
    "marital-status": "marital_status",
    "occupation": "occupation",
    "relationship": "relationship",
    "race": "race",
    "sex": "sex",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "hours-per-week": "hours_per_week",
    "native-country": "native_country",
    "income": "income",  # should be two values: <=$50K, >$50K
}


def clean_columns(df: pd.DataFrame, renaming_map: dict) -> pd.DataFrame:
    """
    Standardizes column names to lowercase with underscores and remove
    redundant final_weight column.
    """
    df = df.rename(columns=renaming_map)

    if "final_weight" in df.columns:
        df = df.drop(columns=["final_weight"])

    return df


def clean_and_binarize_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'income' column by removing trailing periods and
    converts it into a binary field called 'high_income' (1 for '>50K', 0 for '=<50K').
    """
    income_col = "income"
    high_income_col = "high_income"

    df[income_col] = df[income_col].astype(str).str.strip(".").str.strip()
    df[high_income_col] = (df[income_col] == ">50K").astype(int)

    return df


def replace_question_marks_with_nan(df: pd.DataFrame, renaming_map: dict):
    """
    Looks for the string '?' in the columns defined by the values of the
    provided COLUMN_RENAMING dictionary and converts those values to np.nan.
    """
    target_cols = [col for col in renaming_map.values() if col in df.columns]
    df[target_cols] = df[target_cols].replace("?", np.nan)

    return df


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


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function that runs all cleaning steps in a logical order.
    """
    df = clean_columns(df, COLUMN_RENAMING)
    df = trim_dataframe_whitespace(df)
    df = replace_question_marks_with_nan(df, COLUMN_RENAMING)
    df = clean_and_binarize_income(df)

    return df
