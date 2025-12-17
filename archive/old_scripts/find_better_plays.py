"""Find better representative plays for DACS visualization"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load supplementary data
supp = pd.read_csv('analytics/data/supplementary_data.csv', low_memory=False)
supp = supp[['game_id', 'play_id', 'pass_result', 'week']].drop_duplicates()
supp = supp[supp['week'] <= 9]  # Only weeks 1-9 have tracking

outcome_map = {
    'C': 'Completion', 'Complete': 'Completion',
    'I': 'Incomplete', 'Incomplete': 'Incomplete',
    'IN': 'Interception', 'Interception': 'Interception'
}

dacs_dir = Path('analytics/outputs/dacs_final_full')
play_files = list(dacs_dir.glob('**/game_*_play_*.json'))

candidates = {
    'rising_dacs_interception': [],
    'low_dacs_completion': [],
    'dacs_battle_incomplete': []
}

print(f"Analyzing {len(play_files)} plays for better representatives...")

for i, pf in enumerate(play_files):
    if i % 1000 == 0:
        print(f"  {i}/{len(play_files)}...")

    with open(pf, 'r') as f:
        data = json.load(f)

    game_id = data['game_id']
    play_id = data['play_id']

    outcome_row = supp[(supp['game_id'] == game_id) & (supp['play_id'] == play_id)]
    if outcome_row.empty:
        continue

    outcome = outcome_map.get(outcome_row['pass_result'].iloc[0], None)
    if not outcome:
        continue

    dacs_series = data.get('dacs_series', [])
    if not dacs_series or len(dacs_series) < 15:  # Need at least 1.5 seconds
        continue

    max_dacs = max(dacs_series)
    min_dacs = min(dacs_series)
    avg_dacs = np.mean(dacs_series)
    final_dacs = dacs_series[-1]
    initial_dacs = dacs_series[0]
    num_frames = len(dacs_series)
    dacs_range = max_dacs - min_dacs

    play_info = {
        'game_id': game_id,
        'play_id': play_id,
        'file': pf,
        'initial_dacs': initial_dacs,
        'max_dacs': max_dacs,
        'min_dacs': min_dacs,
        'avg_dacs': avg_dacs,
        'final_dacs': final_dacs,
        'num_frames': num_frames,
        'dacs_range': dacs_range,
        'score': 0
    }

    # Rising DACS Interception: Shows DACS increasing as defenders converge
    if outcome == 'Interception' and num_frames >= 15:
        if dacs_range > 20 and final_dacs > 40:
            # Score: favor rise + good length + high final
            rise_score = (final_dacs - initial_dacs) * 0.5 + num_frames * 0.3 + final_dacs * 0.2
            play_info['score'] = rise_score
            candidates['rising_dacs_interception'].append(play_info)

    # Low DACS Completion: Low DACS shows defense can't control
    if outcome == 'Completion' and num_frames >= 15:
        if avg_dacs < 20 and max_dacs < 40:
            # Score: favor low average + good length
            consistency_score = (40 - avg_dacs) * 0.5 + num_frames * 0.3
            play_info['score'] = consistency_score
            candidates['low_dacs_completion'].append(play_info)

    # DACS Battle Incomplete: Moderate DACS, shows competition
    if outcome == 'Incomplete' and num_frames >= 15:
        if dacs_range > 15 and 20 < avg_dacs < 60:
            # Score: favor variance + good length + moderate final
            battle_score = dacs_range * 0.4 + num_frames * 0.3 + (100 - abs(40 - avg_dacs)) * 0.2
            play_info['score'] = battle_score
            candidates['dacs_battle_incomplete'].append(play_info)

# Sort and select best
results = {}
for category, plays in candidates.items():
    if plays:
        sorted_plays = sorted(plays, key=lambda x: x['score'], reverse=True)
        print(f"\n{'='*70}")
        print(f"{category.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        for i, play in enumerate(sorted_plays[:5]):
            print(f"\n{i+1}. Game: {play['game_id']}, Play: {play['play_id']}")
            print(f"   Frames: {play['num_frames']}, Score: {play['score']:.1f}")
            print(f"   DACS: {play['initial_dacs']:.1f}% -> {play['final_dacs']:.1f}% (range: {play['dacs_range']:.1f}%)")
            print(f"   Avg: {play['avg_dacs']:.1f}%, Max: {play['max_dacs']:.1f}%")
