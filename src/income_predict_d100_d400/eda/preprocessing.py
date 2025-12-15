from typing import Any, Dict, List

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

from income_predict_d100_d400.eda import plots as plotting


def get_data_description(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Generates a description of the data and plots distributions.

    Parameters:
        df: The DataFrame to analyze.

    Returns:
        A dictionary containing data types and a dataframe of descriptive statistics.
    """
    plotting.plot_distributions(df)

    return {"dtypes": df.schema, "description": df.describe()}


def get_target_distribution(df: pl.DataFrame, target: str = "income") -> pl.DataFrame:
    """
    Calculates the distribution of the target variable.

    Parameters:
        df: The DataFrame containing the target column.
        target: The name of the target column.

    Returns:
        A DataFrame with the count and percentage distribution of the target.
    """
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in dataframe.")

    dist_df = (
        df.group_by(target)
        .len()
        .with_columns((pl.col("len") / df.height * 100).alias("Percent"))
        .rename({"len": "Count"})
        .sort(target)
    )

    plotting.plot_target_distribution(dist_df, target)

    return dist_df


def get_outliers_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Identifies outliers in numeric columns using the IQR method and checks for missing values.

    Parameters:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame summarizing outlier counts, bounds, and missing values for each column.
    """
    numeric_cols = [col for col, dtype in df.schema.items() if dtype.is_numeric()]
    non_numeric_cols = [
        col for col, dtype in df.schema.items() if not dtype.is_numeric()
    ]
    outlier_summary = []

    for col in numeric_cols:
        missing_count = df.select(pl.col(col).null_count()).item()

        # Calculate IQR
        quantiles = df.select(
            [
                pl.col(col).quantile(0.25).alias("q1"),
                pl.col(col).quantile(0.75).alias("q3"),
            ]
        ).row(0)

        Q1, Q3 = quantiles
        if Q1 is None or Q3 is None:
            continue

        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        outlier_count = df.filter(
            (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
        ).height

        if outlier_count > 0 or missing_count > 0:
            outlier_summary.append(
                {
                    "Column": col,
                    "Outlier Count": outlier_count,
                    "Percent": (outlier_count / df.height) * 100,
                    "Lower Bound": lower_bound,
                    "Upper Bound": upper_bound,
                    "Missing Values": missing_count,
                }
            )

    for col in non_numeric_cols:
        missing_count = df.select(pl.col(col).null_count()).item()

        if missing_count > 0:
            outlier_summary.append(
                {
                    "Column": col,
                    "Outlier Count": None,
                    "Percent": None,
                    "Lower Bound": None,
                    "Upper Bound": None,
                    "Missing Values": missing_count,
                }
            )

    if not outlier_summary:
        return pl.DataFrame(
            schema={
                "Column": pl.String,
                "Outlier Count": pl.Int64,
                "Percent": pl.Float64,
                "Lower Bound": pl.Float64,
                "Upper Bound": pl.Float64,
                "Missing Values": pl.Int64,
            }
        )

    plotting.plot_numeric_boxplots(df)

    return pl.DataFrame(outlier_summary).sort("Percent", descending=True)


def calculate_categorical_correlations(
    df: pl.DataFrame, target: str = "income"
) -> pl.DataFrame:
    """
    Calculate Cramér's V correlation between categorical features and the target.

    Parameters:
        df: The DataFrame containing categorical features.
        target: The name of the target column.

    Returns:
        A DataFrame of Cramér's V values for each categorical feature.
    """
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in dataframe.")

    categorical_cols = [
        col
        for col, dtype in df.schema.items()
        if dtype in (pl.Categorical, pl.String, pl.Boolean) and col != target
    ]

    results = []

    for col in categorical_cols:
        pivot = df.pivot(
            index=col, on=target, values=target, aggregate_function="len"
        ).fill_null(0)

        contingency_table = pivot.select(pl.all().exclude(col)).to_numpy()

        chi2, _, _, _ = chi2_contingency(contingency_table)

        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

        results.append({"feature": col, "correlation": cramers_v})

    return pl.DataFrame(results).sort("correlation", descending=True)


def get_feature_correlations(
    df: pl.DataFrame, target: str = "income"
) -> Dict[str, pl.DataFrame]:
    """
    Calculates Pearson correlation for numeric features and Cramér's V for categorical features.

    Parameters:
        df: The DataFrame to analyze.
        target: The target variable name.

    Returns:
        A dictionary containing two DataFrames: 'numeric' correlations and 'categorical'
        correlations.
    """
    results: Dict[str, pl.DataFrame] = {}

    df_temp = df.clone()
    target_dtype = df.schema[target]

    if not target_dtype.is_numeric():
        if target_dtype == pl.Boolean:
            df_temp = df_temp.with_columns(pl.col(target).cast(pl.Int8).alias(target))
        else:
            df_temp = df_temp.with_columns(
                pl.col(target).cast(pl.Categorical).to_physical().alias(target)
            )

    numeric_cols = [col for col, dtype in df_temp.schema.items() if dtype.is_numeric()]

    corr_matrix = df_temp.select(numeric_cols).corr()

    if target in corr_matrix.columns:
        correlations = corr_matrix[target].to_list()
        numeric_corr_df = (
            pl.DataFrame({"feature": numeric_cols, "correlation": correlations})
            .filter(pl.col("feature") != target)
            .sort("correlation", descending=True)
        )
    else:
        numeric_corr_df = pl.DataFrame(
            schema={"feature": pl.String, "correlation": pl.Float64}
        )

    results["numeric"] = numeric_corr_df

    results["categorical"] = calculate_categorical_correlations(df, target)

    plotting.plot_numeric_strip(df, target)
    plotting.plot_feature_correlations(
        results["numeric"], target, title_suffix="(Numeric - Pearson)"
    )
    plotting.plot_categorical_stack(df, target)
    plotting.plot_feature_correlations(
        results["categorical"], target, title_suffix="(Categorical - Cramér's V)"
    )

    return results


def identify_features_by_type(df: pl.DataFrame) -> Dict[str, List[str]]:
    """
    Separates available columns into 'numeric' and 'categorical' lists.

    Parameters:
        df: The DataFrame to examine.

    Returns:
        A dictionary with keys 'numeric' and 'categorical', containing lists of column names.
    """
    target_col = "high_income"
    features = df.drop([target_col]) if target_col in df.columns else df

    return {
        "numeric": [
            col for col, dtype in features.schema.items() if dtype.is_numeric()
        ],
        "categorical": [
            col
            for col, dtype in features.schema.items()
            if dtype in (pl.Categorical, pl.String, pl.Boolean)
        ],
    }
