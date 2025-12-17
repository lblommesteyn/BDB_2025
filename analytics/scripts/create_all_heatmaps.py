"""
Create multiple heatmap versions using different DACS metrics
- Average DACS
- Median DACS
- Max DACS
- Final DACS (at ball arrival)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

def load_all_play_data():
    """Load all play data from the full dataset"""
    dacs_dir = Path('analytics/outputs/dacs_final_full')
    if not dacs_dir.exists():
        dacs_dir = Path('analytics/outputs/dacs')

    play_files = list(dacs_dir.glob('**/game_*_play_*.json'))
    print(f"Loading from: {dacs_dir}")
    print(f"Found {len(play_files)} play files")

    play_data = []
    for i, pf in enumerate(play_files):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(play_files)}...")

        with open(pf, 'r') as f:
            data = json.load(f)

            if 'corridor_centers' in data and len(data['corridor_centers']) > 0:
                final_pos = data['corridor_centers'][-1]
                dacs_series = data.get('dacs_series', [])

                if dacs_series:
                    play_data.append({
                        'play_id': data['play_id'],
                        'ball_x': final_pos[0],
                        'ball_y': final_pos[1],
                        'corridor_length': data.get('corridor_length', 0),
                        'max_dacs': max(dacs_series),
                        'avg_dacs': np.mean(dacs_series),
                        'median_dacs': np.median(dacs_series),
                        'final_dacs': dacs_series[-1] if dacs_series else 0,
                        'n_defenders': data.get('n_defenders', 0)
                    })

    print(f"Loaded {len(play_data)} plays with DACS data")
    return pd.DataFrame(play_data)

def create_heatmap(df, metric_col, metric_name, output_name, output_dir='analytics/outputs/presentation'):
    """Create a heatmap for a specific DACS metric"""

    # Categorize plays
    df['depth_category'] = pd.cut(df['corridor_length'],
                                   bins=[0, 5, 15, 25, 100],
                                   labels=['Screen/Short', 'Medium', 'Deep', 'Very Deep'])

    df['lateral_category'] = pd.cut(df['ball_y'],
                                     bins=[-10, 18, 35, 60],
                                     labels=['Left', 'Middle', 'Right'])

    # Create heatmap data
    heatmap = df.groupby(['depth_category', 'lateral_category'], observed=False).agg({
        metric_col: 'mean',
        'play_id': 'count'
    }).round(1)

    heatmap_values = heatmap[metric_col].unstack(fill_value=0)
    counts = heatmap['play_id'].unstack(fill_value=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(13, 9))

    # Determine color scale
    vmax = df[metric_col].quantile(0.90)  # Use 90th percentile to avoid outliers

    sns.heatmap(heatmap_values,
                annot=True, fmt='.1f',
                cmap='RdYlGn',
                vmin=0,
                vmax=vmax,
                center=df[metric_col].median(),
                cbar_kws={'label': f'{metric_name}\n(Higher = Better Defense)', 'pad': 0.02},
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
    ax.set_title(f'{metric_name} by Field Zone\nPost-Throw Ball Flight Analysis (9,852 Plays)',
                fontsize=17, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Field Position (Left-Right)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Pass Depth from Line of Scrimmage', fontsize=14, fontweight='bold', color='#2c3e50')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    # Add note
    note_text = (
        f'Based on 9,852 plays from 2023 season. {metric_name} calculated across all frames during ball flight.\n'
        'Green zones = higher defensive control. Red/yellow = offensive advantage.'
    )
    fig.text(0.5, 0.02, note_text,
             ha='center', fontsize=11, style='italic',
             color='#7f8c8d', wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save
    output_path = Path(output_dir) / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved {metric_name} heatmap to {output_path}")
    plt.close()

    # Print stats
    print(f"\n{metric_name} Stats:")
    print(f"  Overall: min={df[metric_col].min():.1f}, median={df[metric_col].median():.1f}, max={df[metric_col].max():.1f}")
    print(f"  By depth:")
    print(df.groupby('depth_category', observed=False)[metric_col].agg(['mean', 'median', 'count']))

if __name__ == '__main__':
    print("="*60)
    print("Creating All Heatmap Versions")
    print("="*60)

    # Load data once
    df = load_all_play_data()

    print(f"\n{'='*60}")
    print("Generating Heatmaps...")
    print("="*60)

    # Create different versions
    create_heatmap(df, 'avg_dacs', 'Average DACS', 'viz_heatmap_avg_dacs.png')
    create_heatmap(df, 'median_dacs', 'Median DACS', 'viz_heatmap_median_dacs.png')
    create_heatmap(df, 'max_dacs', 'Peak DACS', 'viz_heatmap_max_dacs.png')
    create_heatmap(df, 'final_dacs', 'Final DACS (At Arrival)', 'viz_heatmap_final_dacs.png')

    print(f"\n{'='*60}")
    print("All heatmaps created!")
    print("="*60)
