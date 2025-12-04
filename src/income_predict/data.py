import argparse
import itertools
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_census_data() -> pd.DataFrame:
    """Fetches the dataset from UCI with a loading animation."""
    container: dict[str, Optional[Any]] = {"data": None, "error": None}

    def _download():
        try:
            container["data"] = fetch_ucirepo(id=2)
        except Exception as e:
            container["error"] = e

    thread = threading.Thread(target=_download)
    thread.start()

    spinner = itertools.cycle([".", "..", "..."])
    max_width = len("⬇️  Downloading... Done!")

    sys.stdout.write("\033[?25l")  # Hide cursor
    try:
        while thread.is_alive():
            message = f"\r⬇️  Downloading{next(spinner)}"
            sys.stdout.write(message.ljust(max_width))
            sys.stdout.flush()
            time.sleep(0.5)

        thread.join()

        if container["error"]:
            sys.stdout.write("\r" + " " * max_width + "\r")
            raise container["error"]

        sys.stdout.write("\r⬇️  Downloading... Done!\n")
        sys.stdout.flush()
    finally:
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.flush()

    data = container["data"]

    if data is None:
        raise RuntimeError("Failed to fetch data from UCI repository")

    if data.data.original is not None:
        return data.data.original

    return pd.concat([data.data.features, data.data.targets], axis=1)


def save_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    file_name: str,
    output_format: Literal["csv", "parquet"],
) -> Path:
    """Saves the dataframe to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{file_name}.{output_format}"

    if output_format == "csv":
        df.to_csv(full_path, index=False)
    else:
        df.to_parquet(full_path, index=False)

    return full_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and save the Census Income dataset."
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=["csv", "parquet"],
        help="Format to save data (default: parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save the output file (default: data)",
    )
    parser.add_argument(
        "--file-name",
        default="census_income",
        help="Name of the output file without extension (default: census_income)",
    )

    args = parser.parse_args()

    try:
        df = fetch_census_data()
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)

    saved_path = save_dataframe(
        df=df,
        output_dir=args.output_dir,
        file_name=args.file_name,
        output_format=args.format,
    )
    print(f"✅ Saved {df.shape[0]} rows, {df.shape[1]} columns to: {saved_path}")


if __name__ == "__main__":
    main()
