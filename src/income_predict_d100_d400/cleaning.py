from typing import Dict, List

import polars as pl

from income_predict_d100_d400.robust_paths import DATA_DIR

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

# Removed "marital_status" from this list so we can process it first
COLUMNS_TO_DROP: List[str] = ["fnlwgt", "education-num", "income"]

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


def encode_education(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert education to ordinal numeric values.

    Parameters:
        df: The DataFrame containing the 'education' column.

    Returns:
        The DataFrame with the 'education' column mapped to integers.
    """
    return df.with_columns(
        pl.col("education").replace_strict(EDUCATION_ORDER, default=None).cast(pl.Int64)
    )


def combine_capital(df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine capital_gain and capital_loss into single capital_net column.

    Parameters:
        df: The DataFrame containing 'capital_gain' and 'capital_loss'.

    Returns:
        The DataFrame with a new 'capital_net' column and original columns removed.
    """
    return df.with_columns(
        (pl.col("capital_gain") - pl.col("capital_loss")).alias("capital_net")
    ).drop(["capital_gain", "capital_loss"])


def combine_married(df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine Husband and Wife into Married in relationship column.

    Parameters:
        df: The DataFrame containing the 'relationship' column.

    Returns:
        The DataFrame with updated 'relationship' values.
    """
    return df.with_columns(
        pl.col("relationship").replace({"Husband": "Married", "Wife": "Married"})
    )


def binarize_race(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert race column to binary columns: is_white and is_black.

    Parameters:
        df: The DataFrame containing the 'race' column.

    Returns:
        The DataFrame with 'is_white' and 'is_black' columns added and 'race' removed.
    """
    return df.with_columns(
        (pl.col("race") == "White").alias("is_white"),
        (pl.col("race") == "Black").alias("is_black"),
    ).drop("race")


def binarize_sex(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert sex column to binary column: is_female.

    Parameters:
        df: The DataFrame containing the 'sex' column.

    Returns:
        The DataFrame with 'is_female' column added and 'sex' removed.
    """
    return df.with_columns((pl.col("sex") == "Female").alias("is_female")).drop("sex")


def binarize_marital_status(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert marital_status column to binary column: is_married_healthy.
    1 if 'Married-civ-spouse' or 'Married-AF-spouse', 0 otherwise.

    Parameters:
        df: The DataFrame containing the 'marital_status' column.

    Returns:
        The DataFrame with 'is_married_healthy' column added and 'marital_status' removed.
    """
    return df.with_columns(
        pl.col("marital_status")
        .is_in(["Married-civ-spouse", "Married-AF-spouse"])
        .alias("is_married_healthy")
    ).drop("marital_status")


def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add interaction features for GLM modeling.

    Parameters:
        df: The DataFrame containing 'age' and 'education'.

    Returns:
        The DataFrame with a new 'age_x_education' column.
    """
    return df.with_columns(
        (pl.col("age") * pl.col("education")).alias("age_x_education")
    )


def add_unique_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a unique_id column to the dataframe as the first column.

    Parameters:
        df: The input DataFrame.

    Returns:
        The DataFrame with a 'unique_id' column inserted at index 0.
    """
    return df.with_row_index(name="unique_id")


def clean_columns(
    df: pl.DataFrame,
    renaming_map: Dict[str, str] = COLUMN_RENAMING,
    columns_to_drop: List[str] = COLUMNS_TO_DROP,
) -> pl.DataFrame:
    """
    Renames a standard set of columns to use snake_case and drops predefined columns.

    Parameters:
        df: The input DataFrame.
        renaming_map: A dictionary mapping old column names to new ones.
        columns_to_drop: A list of column names to remove.

    Returns:
        The cleaned DataFrame with renamed columns and dropped features.
    """
    columns_to_drop_in_df = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop_in_df:
        df = df.drop(columns_to_drop_in_df)

    return df.rename(renaming_map)


def clean_and_binarize_income(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans the 'income' column and converts it into a boolean field.

    Parameters:
        df: The DataFrame containing the 'income' column.

    Returns:
        The DataFrame with a cleaned 'income' column and a new 'high_income' boolean column.
    """
    return df.with_columns(
        pl.col("income").cast(pl.String).str.strip_chars().str.strip_chars(".")
    ).with_columns((pl.col("income") == ">50K").alias("high_income"))


def replace_question_marks_with_nan(df: pl.DataFrame) -> pl.DataFrame:
    """
    Replaces '?' (plus any whitespace e.g. ' ?') with null across all columns in the dataframe.

    Parameters:
        df: The input DataFrame.

    Returns:
        The DataFrame with '?' strings replaced by nulls.
    """
    return df.with_columns(pl.col(pl.String).str.strip_chars().replace("?", None))


def trim_dataframe_whitespace(df: pl.DataFrame) -> pl.DataFrame:
    """
    Automatically detects string columns and strips whitespace from all values.

    Parameters:
        df: The input DataFrame.

    Returns:
        The DataFrame with whitespace stripped from all string columns.
    """
    return df.with_columns(pl.col(pl.String).str.strip_chars())


def full_clean(df: pl.DataFrame) -> pl.DataFrame:
    """
    Master function that runs all cleaning steps in a logical order.

    Parameters:
        df: The raw input DataFrame.

    Returns:
        The fully cleaned and preprocessed DataFrame.
    """
    df = add_unique_id(df)
    df = clean_and_binarize_income(df)
    df = clean_columns(df, COLUMN_RENAMING)
    df = trim_dataframe_whitespace(df)
    df = replace_question_marks_with_nan(df)
    df = encode_education(df)
    df = combine_capital(df)
    df = combine_married(df)
    df = binarize_race(df)
    df = binarize_sex(df)
    df = binarize_marital_status(df)  # Added new step
    df = add_interaction_features(df)

    return df


def run_cleaning_pipeline(df: pl.DataFrame) -> None:
    """
    Runs full cleaning pipeline and saves result in parquet format.

    Parameters:
        df: The raw input DataFrame.
    """
    df = full_clean(df)

    output_path = DATA_DIR / "cleaned_census_income.parquet"

    df.write_parquet(output_path)
