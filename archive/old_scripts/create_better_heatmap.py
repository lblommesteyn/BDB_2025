"""
Create a better heatmap that shows MEANINGFUL differences
Instead of absolute DACS (mostly 0), show:
1. Relative DACS (normalized)
2. Coverage difficulty score
3. Success rate by zone
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

def create_coverage_difficulty_map(output_dir='analytics/outputs/presentation'):
    """
    Create a heatmap showing coverage difficulty by zone
    Uses completion rate and average yards instead of raw DACS
    """

    # Load play data
    play_files = list(Path('analytics/outputs/dacs').glob('game_*_play_*.json'))[:50]

    play_data = []
    for pf in play_files:
        with open(pf, 'r') as f:
            data = json.load(f)
            if 'corridor_centers' in data and len(data['corridor_centers']) > 0:
                final_pos = data['corridor_centers'][-1]
                max_dacs = max(data['dacs_series']) if data['dacs_series'] else 0
                avg_dacs = np.mean(data['dacs_series']) if data['dacs_series'] else 0

                play_data.append({
                    'play_id': data['play_id'],
                    'ball_x': final_pos[0],
                    'ball_y': final_pos[1],
                    'max_dacs': max_dacs,
                    'avg_dacs': avg_dacs,
                    'corridor_length': data.get('corridor_length', 0)
                })

    if not play_data:
        print("No play data found")
        return

    play_df = pd.DataFrame(play_data)

    # Load supplementary for outcomes
    supp = pd.read_csv('analytics/data/supplementary_data.csv', low_memory=False)
    supp_merge = supp[['game_id', 'play_id', 'pass_result']].drop_duplicates()

    # For now, categorize by field position
    # Depth categories (from LOS)
    play_df['depth_zone'] = pd.cut(play_df['corridor_length'],
                                    bins=[0, 10, 20, 35, 100],
                                    labels=['Short (<10)', 'Medium (10-20)',
                                           'Deep (20-35)', 'Very Deep (35+)'])

    # Lateral categories (field position)
    play_df['lateral_zone'] = pd.cut(play_df['ball_y'],
                                      bins=[-10, 17.67, 35.33, 60],
                                      labels=['Left Third', 'Middle Third', 'Right Third'])

    # Calculate coverage difficulty score
    # Higher DACS = easier to defend = lower difficulty
    # So difficulty = 100 - avg_dacs
    play_df['coverage_difficulty'] = 100 - play_df['avg_dacs']

    # Also calculate a "defensive success score"
    # Combines max DACS with consistency
    play_df['def_success'] = play_df['max_dacs'] * 0.6 + play_df['avg_dacs'] * 0.4

    # Create heatmap data
    heatmap_data = play_df.groupby(['depth_zone', 'lateral_zone'])['def_success'].mean().unstack(fill_value=0)

    # Also get counts for annotation
    count_data = play_df.groupby(['depth_zone', 'lateral_zone']).size().unstack(fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a diverging colormap centered at median
    median_val = play_df['def_success'].median()

    sns.heatmap(heatmap_data, annot=True, fmt='.1f',
                cmap='RdYlGn', center=median_val,
                vmin=0, vmax=play_df['def_success'].max(),
                cbar_kws={'label': 'Defensive Success Score\n(Higher = Better Control)'},
                linewidths=3, linecolor='white',
                annot_kws={'fontsize': 13, 'fontweight': 'bold'},
                ax=ax)

    # Add sample sizes as text
    for i, depth in enumerate(heatmap_data.index):
        for j, lateral in enumerate(heatmap_data.columns):
            count = count_data.loc[depth, lateral] if depth in count_data.index and lateral in count_data.columns else 0
            if count > 0:
                ax.text(j + 0.5, i + 0.75, f'(n={int(count)})',
                       ha='center', va='top', fontsize=9, style='italic',
                       color='gray')

    ax.set_title('Defensive Success by Field Zone\n(Higher = Defense Controls Space Better)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Field Position (Horizontal)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pass Depth from Line of Scrimmage', fontsize=13, fontweight='bold')

    # Add interpretation note
    fig.text(0.5, 0.02,
             'Note: Defensive Success Score combines peak and average DACS during ball flight. '
             'Green zones = easier to defend, Red zones = offensive advantage',
             ha='center', fontsize=10, style='italic', wrap=True)

    plt.tight_layout()

    output_path = Path(output_dir) / 'viz_route_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved improved heatmap to {output_path}")
    plt.close()

    # Print stats
    print(f"\nStats:")
    print(f"Total plays: {len(play_df)}")
    print(f"Avg defensive success by depth:")
    print(play_df.groupby('depth_zone')['def_success'].mean())

if __name__ == '__main__':
    create_coverage_difficulty_map()
