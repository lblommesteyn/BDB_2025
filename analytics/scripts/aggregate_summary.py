import os
import pandas as pd
import glob
from concurrent.futures import ThreadPoolExecutor

def load_game_summary(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main():
    root_dir = r"c:\Users\16476\BDB_2025"
    outputs_dir = os.path.join(root_dir, "analytics", "outputs", "dacs_final_full")
    summary_path = os.path.join(root_dir, "season_summary.parquet")
    
    print(f"Looking for game summaries in {outputs_dir}...")
    # Find all game summary CSVs
    # Pattern: game_*/game_*_dacs_summary.csv
    search_pattern = os.path.join(outputs_dir, "game_*", "*_dacs_summary.csv")
    files = glob.glob(search_pattern)
    
    print(f"Found {len(files)} summary files.")
    
    dfs = []
    # Use threading to speed up reading
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_game_summary, files)
        for res in results:
            if res is not None:
                dfs.append(res)
                
    if not dfs:
        print("No data found.")
        return
        
    print(f"Concatenating {len(dfs)} dataframes...")
    full_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Saving to {summary_path}...")
    full_df.to_parquet(summary_path, index=False)
    print("Done.")
    print(f"Total rows: {len(full_df)}")

if __name__ == "__main__":
    main()
