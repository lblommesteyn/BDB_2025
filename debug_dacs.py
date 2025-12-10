import pandas as pd
import os

path = r'c:\Users\16476\BDB_2025\analytics\data\raw\114239_nfl_competition_files_published_analytics_final\train\input_2023_w01.csv'
target = 2023090700

try:
    for chunk in pd.read_csv(path, chunksize=200_000, dtype={'player_side': str}):
        print(f"Chunk player_side unique: {chunk['player_side'].unique()[:5]}")
        g = chunk[chunk['game_id'] == target]
        if not g.empty:
            print(f"Found {len(g)} rows")
            print(f"Sides: {g['player_side'].value_counts(dropna=False)}")
            break
except Exception as e:
    print(f"Error reading CSV: {e}")
