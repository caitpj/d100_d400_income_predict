import sys
from pathlib import Path

import pandas as pd

current_file = Path(__file__).resolve()
src_directory = current_file.parent.parent
sys.path.append(str(src_directory))

from income_predict.cleaning import run_cleaning_pipeline
from income_predict.data import run_data_fetch_pipeline

print("Starting Pipeline...")

file_path = run_data_fetch_pipeline()
df_raw = pd.read_parquet(file_path)
run_cleaning_pipeline(df_raw)

print("Pipeline finished.")
