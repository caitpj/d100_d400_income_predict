import itertools
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_census_data() -> pd.DataFrame:
    """
    Fetches the dataset from UCI with a loading animation.

    Returns:
        A DataFrame containing the merged features and targets from the dataset.
    """
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


def load_data(output_format: Literal["csv", "parquet"] = "parquet") -> Path:
    """
    Checks if data exists locally. If not, downloads Census Income dataset.

    Parameters:
        output_format: The format to save the data in ('csv' or 'parquet').

    Returns:
        The Path to the saved dataset file.
    """
    if (Path.cwd() / "src").exists():
        data_dir = Path.cwd() / "src" / "data"
    else:
        data_dir = Path.cwd() / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    file_name = "census_income"
    output_path = data_dir / f"{file_name}.{output_format}"

    if output_path.exists():
        print(
            f"""
No need to download since data already exists at:
{data_dir.name}/{output_path.name} (good, because it means
we're less likley to get blocked by UCI for spamming downloads)
            """
        )
        return output_path

    try:
        df = fetch_census_data()
    except Exception as e:
        raise RuntimeError(f"Error downloading data: {e}") from e

    if output_format == "csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    return output_path
