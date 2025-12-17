"""
Create improved visualizations for BDB 2026 submission
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Use clean style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_improved_evolution_chart(output_dir='analytics/outputs/presentation'):
    """Create DACS evolution chart focusing on meaningful plays."""

    # Load data
    ts = pd.read_parquet('analytics/outputs/dacs/game_2023090700_dacs_timeseries.parquet')
    supp = pd.read_csv('analytics/data/supplementary_data.csv')
    supp = supp[['game_id', 'play_id', 'pass_result']].drop_duplicates()

    # Merge
    merged = ts.merge(supp, on=['game_id', 'play_id'], how='left')
    merged = merged.dropna(subset=['pass_result'])

    # Map outcomes
    outcome_map = {
        'C': 'Completion', 'Complete': 'Completion',
        'I': 'Incomplete', 'Incomplete': 'Incomplete',
        'IN': 'Interception', 'Interception': 'Interception'
    }
    merged['outcome'] = merged['pass_result'].map(outcome_map)
    merged = merged.dropna(subset=['outcome'])

    # Filter to plays with meaningful DACS (at least one frame > 5%)
    play_max_dacs = merged.groupby('play_id')['dacs'].max()
    meaningful_plays = play_max_dacs[play_max_dacs > 5].index
    merged = merged[merged['play_id'].isin(meaningful_plays)]

    print(f"Filtered to {len(meaningful_plays)} plays with meaningful DACS")
    print(f"Outcomes: {merged.groupby('outcome')['play_id'].nunique().to_dict()}")

    # Normalize time
    def normalize_time(group):
        t_max = group['t'].max()
        if t_max > 0:
            group['t_norm'] = group['t'] / t_max
        else:
            group['t_norm'] = 0
        return group

    merged = merged.groupby('play_id', group_keys=False, as_index=False).apply(normalize_time)

    # Create time bins
    time_bins = np.linspace(0, 1, 11)
    merged['time_bin'] = pd.cut(merged['t_norm'], bins=time_bins, labels=False, include_lowest=True)

    # Statistics
    summary = merged.groupby(['outcome', 'time_bin'])['dacs'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = {
        'Interception': '#e74c3c',
        'Incomplete': '#3498db',
        'Completion': '#2ecc71'
    }

    markers = {'Interception': 'o', 'Incomplete': 's', 'Completion': '^'}

    for outcome in ['Interception', 'Incomplete', 'Completion']:
        data = summary[summary['outcome'] == outcome]
        if len(data) > 0:
            x = data['time_bin'] / 10.0
            y = data['mean']

            ax.plot(x, y, marker=markers[outcome], linewidth=3, label=outcome,
                   color=colors[outcome], markersize=8, alpha=0.9, markeredgewidth=2,
                   markeredgecolor='white')

            # Confidence interval
            if len(data[data['count'] >= 2]) > 0:
                valid = data[data['count'] >= 2]
                x_valid = valid['time_bin'] / 10.0
                y_valid = valid['mean']
                se_valid = valid['se']
                ax.fill_between(x_valid,
                               np.maximum(0, y_valid - 1.96*se_valid),
                               y_valid + 1.96*se_valid,
                               alpha=0.2, color=colors[outcome])

    # Reference lines
    ax.axhline(y=50, color='#7f8c8d', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(0.02, 52, 'Equal Control (50%)', fontsize=11, color='#7f8c8d',
            style='italic', weight='bold')

    # Labels
    ax.set_xlabel('Normalized Ball Flight Time\n(0 = Release, 1 = Arrival)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Defensive Air Control Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('DACS Evolution During Ball Flight by Outcome',
                fontsize=15, fontweight='bold', pad=15)

    # Legend with sample sizes
    legend_labels = []
    for outcome in ['Interception', 'Incomplete', 'Completion']:
        n = merged[merged['outcome'] == outcome]['play_id'].nunique()
        legend_labels.append(f'{outcome} (n={n})')

    ax.legend(legend_labels, loc='upper left', frameon=True, fontsize=12,
             shadow=True, fancybox=True)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add insight annotation
    ax.annotate('Higher DACS predicts\ndefensive success',
                xy=(0.7, 35), xytext=(0.75, 55),
                arrowprops=dict(arrowstyle='->', color='#34495e', lw=2),
                fontsize=11, color='#2c3e50', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#34495e', alpha=0.9, linewidth=2))

    plt.tight_layout()

    output_path = Path(output_dir) / 'viz_dacs_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved improved DACS evolution chart to {output_path}")
    plt.close()


def create_route_heatmap(output_dir='analytics/outputs/presentation'):
    """Create heatmap of DACS by field location."""

    # Load supplementary data for ball landing locations
    supp = pd.read_csv('analytics/data/supplementary_data.csv')

    # Load DACS summary
    summary = pd.read_csv('analytics/outputs/dacs/game_2023090700_dacs_summary.csv')

    # Merge to get pass results
    supp_cols = [c for c in ['game_id', 'play_id', 'pass_result', 'absolute_yardline_number'] if c in supp.columns]
    merged = summary.merge(
        supp[supp_cols],
        on=['game_id', 'play_id'],
        how='left'
    )

    # For heatmap, we need actual field positions
    # Let's load a few play JSONs to get ball_land positions
    import json
    from glob import glob

    play_files = glob('analytics/outputs/dacs/game_*_play_*.json')[:50]

    play_data = []
    for pf in play_files:
        with open(pf, 'r') as f:
            data = json.load(f)
            if 'corridor_centers' in data and len(data['corridor_centers']) > 0:
                # Get final corridor center (ball landing area)
                final_pos = data['corridor_centers'][-1]
                play_data.append({
                    'play_id': data['play_id'],
                    'ball_x': final_pos[0],
                    'ball_y': final_pos[1],
                    'dacs_final': data['dacs_series'][-1] if data['dacs_series'] else 0
                })

    if not play_data:
        print("No play data found for heatmap")
        return

    play_df = pd.DataFrame(play_data)

    # Categorize by field zones
    play_df['zone_x'] = pd.cut(play_df['ball_x'], bins=[0, 30, 60, 90, 120],
                                labels=['Behind LOS', 'Short', 'Intermediate', 'Deep'])
    play_df['zone_y'] = pd.cut(play_df['ball_y'], bins=[-5, 13.33, 26.66, 40, 60],
                                labels=['Left', 'Left-Center', 'Right-Center', 'Right'])

    # Calculate average DACS by zone
    heatmap_data = play_df.groupby(['zone_x', 'zone_y'])['dacs_final'].mean().unstack(fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, center=50,
                cbar_kws={'label': 'Average DACS (%)'},
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_title('Field Vulnerability: Average DACS by Target Zone\n(Higher = Better Defensive Control)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Field Zone (Sideline)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pass Depth', fontsize=13, fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir) / 'viz_route_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved route heatmap to {output_path}")
    plt.close()


if __name__ == '__main__':
    print("Creating improved visualizations...")
    print("\n1. DACS Evolution Chart")
    create_improved_evolution_chart()

    print("\n2. Route Heatmap")
    create_route_heatmap()

    print("\n[OK] All visualizations updated!")
