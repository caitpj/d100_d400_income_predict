import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data(output_format: Literal["csv", "parquet"] = "parquet") -> pd.DataFrame:
    """
    Fetches the Census Income dataset (UCI ID=2) and saves the combined
    dataset to a 'data' folder.

    Args:
        output_format (str): The file format to save as. Options: 'csv', 'parquet'.

    Returns:
        pd.DataFrame: The combined dataset containing features and targets.
    """
    if output_format not in ["csv", "parquet"]:
        raise ValueError("output_format must be either 'csv' or 'parquet'")

    print("Fetching Census Income dataset (UCI ID=2)...")

    # fetch dataset
    dataset_repo = fetch_ucirepo(id=2)

    # Access the combined dataframe directly
    df: pd.DataFrame
    if dataset_repo.data.original is not None:
        df = dataset_repo.data.original
    else:
        print(
            "Warning: 'original' dataframe not found. Merging features and targets manually."
        )
        df = pd.concat([dataset_repo.data.features, dataset_repo.data.targets], axis=1)

    # Define path for the data folder
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving census income data to {data_dir.resolve()} as {output_format}...")

    file_path: Path
    if output_format == "csv":
        file_path = data_dir / "census_income.csv"
        df.to_csv(file_path, index=False)

    elif output_format == "parquet":
        file_path = data_dir / "census_income.parquet"
        # Requires pyarrow to be installed
        df.to_parquet(file_path, index=False)

    print(f"✅ Data saved successfully to: {file_path}")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save the Census Income dataset."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["csv", "parquet"],
        help="Format to save data: 'csv' or 'parquet' (default: parquet)",
    )

    args = parser.parse_args()

    load_data(output_format=args.format)
