import pandas as pd


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a summary of missing values for each column."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    summary = pd.DataFrame({"Missing Values": missing, "Percent": percent})
    return summary[summary["Missing Values"] > 0].sort_values(
        "Percent", ascending=False
    )
