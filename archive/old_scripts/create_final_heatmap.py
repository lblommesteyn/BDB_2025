"""
Create final heatmap using ALL available play data
Shows relative defensive performance by zone
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

def create_final_heatmap(output_dir='analytics/outputs/presentation'):
    """Use ALL play data to create meaningful heatmap"""

    # Load ALL play JSONs from the full dataset
    dacs_dir = Path('analytics/outputs/dacs_final_full')
    if not dacs_dir.exists():
        dacs_dir = Path('analytics/outputs/dacs')  # Fallback

    play_files = list(dacs_dir.glob('**/game_*_play_*.json'))
    print(f"Using directory: {dacs_dir}")
    print(f"Found {len(play_files)} play files")

    play_data = []
    for pf in play_files:
        with open(pf, 'r') as f:
            data = json.load(f)

            # Get final ball position
            if 'corridor_centers' in data and len(data['corridor_centers']) > 0:
                final_pos = data['corridor_centers'][-1]

                # Calculate meaningful metrics
                dacs_series = data.get('dacs_series', [])
                if dacs_series:
                    max_dacs = max(dacs_series)
                    avg_dacs = np.mean(dacs_series)
                    final_dacs = dacs_series[-1] if dacs_series else 0
                else:
                    max_dacs = avg_dacs = final_dacs = 0

                play_data.append({
                    'play_id': data['play_id'],
                    'ball_x': final_pos[0],
                    'ball_y': final_pos[1],
                    'corridor_length': data.get('corridor_length', 0),
                    'max_dacs': max_dacs,
                    'avg_dacs': avg_dacs,
                    'final_dacs': final_dacs,
                    'n_defenders': data.get('n_defenders', 0)
                })

    print(f"Loaded {len(play_data)} plays with data")

    if not play_data:
        print("ERROR: No play data found!")
        return

    df = pd.DataFrame(play_data)

    # Categorize plays by depth and lateral position
    df['depth_category'] = pd.cut(df['corridor_length'],
                                   bins=[0, 5, 15, 25, 100],
                                   labels=['Screen/Short', 'Medium', 'Deep', 'Very Deep'])

    df['lateral_category'] = pd.cut(df['ball_y'],
                                     bins=[-10, 18, 35, 60],
                                     labels=['Left', 'Middle', 'Right'])

    # Calculate "Defensive Effectiveness Score"
    # Combines max DACS (peak control) with average DACS (sustained control)
    # Scale to 0-100 for interpretability
    df['def_effectiveness'] = (df['max_dacs'] * 0.6 + df['avg_dacs'] * 0.4)

    # Create heatmap data
    heatmap = df.groupby(['depth_category', 'lateral_category']).agg({
        'def_effectiveness': 'mean',
        'play_id': 'count'
    }).round(1)

    heatmap_values = heatmap['def_effectiveness'].unstack(fill_value=0)
    counts = heatmap['play_id'].unstack(fill_value=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(13, 9))

    # Use custom colormap - red (offense advantage) to green (defense advantage)
    # Center at the median value
    median_val = df['def_effectiveness'].median()

    sns.heatmap(heatmap_values,
                annot=True, fmt='.1f',
                cmap='RdYlGn',
                center=median_val,
                vmin=0,
                vmax=df['def_effectiveness'].quantile(0.95),
                cbar_kws={'label': 'Defensive Effectiveness\n(Higher = Better Control)', 'pad': 0.02},
                linewidths=4,
                linecolor='white',
                annot_kws={'fontsize': 15, 'fontweight': 'bold'},
                ax=ax)

    # Add sample counts
    for i, depth in enumerate(heatmap_values.index):
        for j, lateral in enumerate(heatmap_values.columns):
            count = counts.loc[depth, lateral] if depth in counts.index and lateral in counts.columns else 0
            if count > 0:
                ax.text(j + 0.5, i + 0.85,
                       f'n={int(count)}',
                       ha='center', va='top',
                       fontsize=10, style='italic',
                       color='#555', alpha=0.8)

    # Styling
    ax.set_title('Defensive Effectiveness by Field Zone\nPost-Throw Ball Flight Analysis (2023 Sample)',
                fontsize=17, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Field Position (Left-Right)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Pass Depth from Line of Scrimmage', fontsize=14, fontweight='bold', color='#2c3e50')

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    # Add interpretation note
    note_text = (
        f'Based on {len(df)} plays. Effectiveness = 60% peak DACS + 40% average DACS during ball flight.\n'
        'Green zones = defense controls space better. Red zones = offensive advantage.'
    )
    fig.text(0.5, 0.02, note_text,
             ha='center', fontsize=11, style='italic',
             color='#7f8c8d', wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save
    output_path = Path(output_dir) / 'viz_route_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved heatmap using {len(df)} plays to {output_path}")
    plt.close()

    # Print stats
    print(f"\nStats:")
    print(f"Total plays: {len(df)}")
    print(f"\nAvg Defensive Effectiveness by Depth:")
    print(df.groupby('depth_category')['def_effectiveness'].agg(['mean', 'count']))
    print(f"\nAvg Defensive Effectiveness by Lateral:")
    print(df.groupby('lateral_category')['def_effectiveness'].agg(['mean', 'count']))
    print(f"\nOverall range: {df['def_effectiveness'].min():.1f} - {df['def_effectiveness'].max():.1f}")
    print(f"Median: {df['def_effectiveness'].median():.1f}")

if __name__ == '__main__':
    create_final_heatmap()
