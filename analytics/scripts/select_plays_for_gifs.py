import pandas as pd
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    if args.summary.endswith('.parquet'):
        df = pd.read_parquet(args.summary)
    else:
        df = pd.read_csv(args.summary)
        
    # Criteria for interesting plays
    plays_to_viz = []
    
    # 1. High DACS Interception (The "Eraser" moment)
    # Filter for Interceptions
    ints = df[df['pass_result'].isin(['IN', 'Interception'])]
    if not ints.empty:
        # Highest DACS final
        best_int = ints.sort_values('dacs_final', ascending=False).iloc[0]
        plays_to_viz.append({
            'game_id': int(best_int['game_id']),
            'play_id': int(best_int['play_id']),
            'desc': 'High DACS Interception'
        })
        
    # 2. Low DACS Completion (The "Busted Coverage")
    comps = df[df['pass_result'].isin(['C', 'Complete'])]
    if not comps.empty:
        worst_comp = comps.sort_values('dacs_final', ascending=True).iloc[0]
        plays_to_viz.append({
            'game_id': int(worst_comp['game_id']),
            'play_id': int(worst_comp['play_id']),
            'desc': 'Low DACS Completion'
        })
        
    # 3. High DACS Incompletion (The "Lockdown")
    incomps = df[df['pass_result'].isin(['I', 'Incomplete'])]
    if not incomps.empty:
        best_incomp = incomps.sort_values('dacs_final', ascending=False).iloc[0]
        plays_to_viz.append({
            'game_id': int(best_incomp['game_id']),
            'play_id': int(best_incomp['play_id']),
            'desc': 'High DACS Incompletion'
        })

    # Save to JSON
    with open(args.out, 'w') as f:
        json.dump(plays_to_viz, f, indent=2)
        
    print(f"Selected {len(plays_to_viz)} plays for visualization.")
    for p in plays_to_viz:
        print(f"- {p['desc']}: Game {p['game_id']} Play {p['play_id']}")

if __name__ == "__main__":
    main()
