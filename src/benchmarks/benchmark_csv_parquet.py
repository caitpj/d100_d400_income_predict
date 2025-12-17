"""
Benchmark comparing CSV vs Parquet for loading, cleaning, and saving census data.
Run multiple iterations for accuracy.

Result:
While Parquet is significantly faster than CSV relativley speaking (around 2x speedup),
the absolute time difference is small for this dataset size. Crucially, Parquet files are
typically much smaller on disk (approx. 10x smaller for this dataset). Given these results,
I selected Parquet files for this project.
"""

import statistics
import sys
import time
from pathlib import Path
from typing import Tuple

import polars as pl

from income_predict_d100_d400.cleaning import full_clean
from income_predict_d100_d400.robust_paths import DATA_DIR

NUM_ITERATIONS: int = 10
DATA_FILE_PATH: Path = DATA_DIR / "census_income.parquet"


def convert_bytes(num: float) -> str:
    """Convert bytes to a human-readable format (KB, MB, GB, etc.)."""
    for unit in ["bytes", "KB", "MB", "GB", "TB", "PB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"


def setup_files() -> Tuple[Path, Path, int, int]:
    """Checks for source data and creates temporary test files."""
    print("--- Setup: Generating Temporary Files ---")

    if not DATA_FILE_PATH.exists():
        print(f"ERROR: Source file not found at {DATA_FILE_PATH}")
        print(
            "Please run 'src/income_predict_d100_d400/pipeline.py' "
            "first to generate the dataset."
        )
        sys.exit(1)

    print(f"Source data found: {DATA_FILE_PATH}")
    df_raw = pl.read_parquet(DATA_FILE_PATH)

    source_csv = Path("temp_csv.csv")
    source_parquet = Path("temp_parquet.parquet")

    print(f"Creating {source_csv}...")
    df_raw.write_csv(source_csv)
    csv_size = source_csv.stat().st_size
    print(f"  -> File size: {convert_bytes(csv_size)}")

    print(f"Creating {source_parquet}...")
    df_raw.write_parquet(source_parquet)
    parquet_size = source_parquet.stat().st_size
    print(f"  -> File size: {convert_bytes(parquet_size)}")

    print("Setup complete.\n")
    return source_csv, source_parquet, csv_size, parquet_size


def run_benchmark_cycle(
    source_file: Path, output_file: Path, format_type: str
) -> float:
    """Runs a single load-clean-save cycle and returns the time taken."""
    start_time = time.perf_counter()

    if format_type == "csv":
        df = pl.read_csv(source_file)
    else:
        df = pl.read_parquet(source_file)

    df = full_clean(df)

    if format_type == "csv":
        df.write_csv(output_file)
    else:
        df.write_parquet(output_file)

    return time.perf_counter() - start_time


def main() -> None:
    source_csv, source_parquet, csv_size, parquet_size = setup_files()
    output_csv = Path("temp_result.csv")
    output_parquet = Path("temp_result.parquet")

    csv_times: list[float] = []
    parquet_times: list[float] = []

    print(f"--- Starting Benchmark ({NUM_ITERATIONS} iterations) ---")

    try:
        for i in range(NUM_ITERATIONS):
            print(f"Iteration {i+1}/{NUM_ITERATIONS}...")

            t_csv = run_benchmark_cycle(source_csv, output_csv, "csv")
            csv_times.append(t_csv)

            t_pq = run_benchmark_cycle(source_parquet, output_parquet, "parquet")
            parquet_times.append(t_pq)

    finally:
        files_to_delete = [source_csv, source_parquet, output_csv, output_parquet]
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    avg_csv = statistics.mean(csv_times)
    avg_pq = statistics.mean(parquet_times)

    print("\n" + "=" * 40)
    print(f"BENCHMARK RESULTS (Average over {NUM_ITERATIONS} runs)")
    print("=" * 40)
    print("⏱️ Time Performance:")
    print(f"  CSV Average Time:     {avg_csv:.4f}s")
    print(f"  Parquet Average Time: {avg_pq:.4f}s")

    if avg_csv < avg_pq:
        speedup = avg_pq / avg_csv
        print(f"  Winner: CSV ({speedup:.2f}x faster)")
    else:
        speedup = avg_csv / avg_pq
        print(f"  Winner: Parquet ({speedup:.2f}x faster)")

    print("-" * 40)
    print("📦 Storage Efficiency:")
    print(f"  CSV File Size:        {convert_bytes(csv_size)}")
    print(f"  Parquet File Size:    {convert_bytes(parquet_size)}")

    if csv_size < parquet_size:
        storage_ratio = parquet_size / csv_size
        print(f"  Winner: CSV ({storage_ratio:.2f}x smaller)")
    else:
        storage_ratio = csv_size / parquet_size
        print(f"  Winner: Parquet ({storage_ratio:.2f}x smaller)")

    print("=" * 40)


if __name__ == "__main__":
    main()
