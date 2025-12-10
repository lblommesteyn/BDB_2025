import pandas as pd
import os

def main():
    # Path to one of the input files
    input_path = r"analytics/data/raw/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv"
    output_path = r"analytics/data/raw/114239_nfl_competition_files_published_analytics_final/players.csv"
    
    print(f"Reading {input_path}...")
    # Read only relevant columns to save memory
    # Based on header: game_id,play_id,player_id,official_position,display_name,...
    # We need player_id and display_name
    
    try:
        # Read just the header
        df_head = pd.read_csv(input_path, nrows=0)
        print(f"Columns found: {list(df_head.columns)}")
        
        # Adjust based on findings
        # Correct columns based on inspection
        # nfl_id, player_name, player_position
        use_cols = []
        rename_map = {}
        
        if 'nfl_id' in df_head.columns:
            use_cols.append('nfl_id')
            rename_map['nfl_id'] = 'nflId'
            
        if 'player_name' in df_head.columns:
            use_cols.append('player_name')
            rename_map['player_name'] = 'displayName'
            
        if 'player_position' in df_head.columns:
            use_cols.append('player_position')
            rename_map['player_position'] = 'position'

        print(f"Using columns: {use_cols}")
        print(f"Rename map: {rename_map}")
        
        df = pd.read_csv(input_path, usecols=use_cols)
        if rename_map:
            df = df.rename(columns=rename_map)
            
        print(f"Columns after rename: {df.columns}")
            
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Extracted {len(df)} rows.")
    print(f"Columns in df: {df.columns}")
    
    # Rename to match standard players.csv format
    if rename_map:
        df = df.rename(columns=rename_map)
        
    # Drop duplicates
    if 'nflId' in df.columns:
        players = df.drop_duplicates(subset=['nflId'])
        # Filter out NaNs (ball)
        players = players.dropna(subset=['nflId'])
        players['nflId'] = players['nflId'].astype(int)
    else:
        print("nflId column not found after rename. Cannot extract players.")
        return
    
    print(f"Found {len(players)} unique players.")
    
    print(f"Saving to {output_path}...")
    players.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
