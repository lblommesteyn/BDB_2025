"""Find alternative high DACS incomplete play from weeks 1-9"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load supplementary data
supp = pd.read_csv('analytics/data/supplementary_data.csv', low_memory=False)
supp = supp[['game_id', 'play_id', 'pass_result', 'week']].drop_duplicates()

# Filter to weeks 1-9
supp = supp[supp['week'] <= 9]

dacs_dir = Path('analytics/outputs/dacs_final_full')
play_files = list(dacs_dir.glob('**/game_*_play_*.json'))

outcome_map = {
    'C': 'Completion', 'Complete': 'Completion',
    'I': 'Incomplete', 'Incomplete': 'Incomplete',
    'IN': 'Interception', 'Interception': 'Interception'
}

candidates = []

print(f"Analyzing {len(play_files)} plays for high DACS incomplete...")

for i, pf in enumerate(play_files):
    if i % 1000 == 0:
        print(f"  {i}/{len(play_files)}...")

    with open(pf, 'r') as f:
        data = json.load(f)

    game_id = data['game_id']
    play_id = data['play_id']

    # Get outcome
    outcome_row = supp[(supp['game_id'] == game_id) & (supp['play_id'] == play_id)]
    if outcome_row.empty:
        continue

    outcome = outcome_map.get(outcome_row['pass_result'].iloc[0], None)
    if outcome != 'Incomplete':
        continue

    dacs_series = data.get('dacs_series', [])
    if not dacs_series or len(dacs_series) < 10:
        continue

    max_dacs = max(dacs_series)
    final_dacs = dacs_series[-1]
    num_frames = len(dacs_series)
    dacs_range = max_dacs - min(dacs_series)

    if final_dacs > 30:
        score = final_dacs * 0.4 + num_frames * 0.3 + dacs_range * 0.3
        candidates.append({
            'game_id': game_id,
            'play_id': play_id,
            'file': pf,
            'max_dacs': max_dacs,
            'final_dacs': final_dacs,
            'num_frames': num_frames,
            'dacs_range': dacs_range,
            'score': score
        })

# Sort by score
candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

print(f"\nTop 5 High DACS Incomplete plays (weeks 1-9):")
for i, play in enumerate(candidates[:5]):
    print(f"{i+1}. Game: {play['game_id']}, Play: {play['play_id']}")
    print(f"   Frames: {play['num_frames']}, Max DACS: {play['max_dacs']:.1f}, Final DACS: {play['final_dacs']:.1f}")
    print(f"   Score: {play['score']:.1f}")
