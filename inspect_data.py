import pandas as pd
import sys

try:
    df = pd.read_parquet(r'c:\Users\16476\BDB_2025\season_summary.parquet')
    print(f"Loaded {len(df)} rows.")
    print("pass_result value counts:")
    print(df['pass_result'].value_counts(dropna=False))
except Exception as e:
    print(f"Error: {e}")
