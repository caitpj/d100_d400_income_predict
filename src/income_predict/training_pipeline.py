import sys
from pathlib import Path

import pandas as pd

current_file = Path(__file__).resolve()
src_directory = current_file.parent.parent
if str(src_directory) not in sys.path:
    sys.path.append(str(src_directory))

from income_predict.cleaning import run_cleaning_pipeline
from income_predict.data import run_data_fetch_pipeline


def main():
    print("Starting Pipeline...")

    file_path = run_data_fetch_pipeline()

    if file_path.suffix == ".parquet":
        df_raw = pd.read_parquet(file_path)
    else:
        df_raw = pd.read_csv(file_path)

    run_cleaning_pipeline(df_raw)

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
