"""
Create DACS Evolution Chart showing how DACS changes during ball flight
for different outcomes (Interception, Incompletion, Completion).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_evolution_chart(output_dir='analytics/outputs/presentation'):
    """Create DACS evolution chart from timeseries and supplementary data."""

    # Load timeseries data
    ts_path = Path('analytics/outputs/dacs/game_2023090700_dacs_timeseries.parquet')
    if not ts_path.exists():
        print(f"Timeseries file not found: {ts_path}")
        return

    ts = pd.read_parquet(ts_path)
    print(f"Loaded {len(ts)} timeseries rows")

    # Load supplementary data for outcomes
    supp_path = Path('analytics/data/supplementary_data.csv')
    if not supp_path.exists():
        print(f"Supplementary data not found: {supp_path}")
        return

    supp = pd.read_csv(supp_path)
    supp = supp[['game_id', 'play_id', 'pass_result']].drop_duplicates()

    # Merge
    merged = ts.merge(supp, on=['game_id', 'play_id'], how='left')
    merged = merged.dropna(subset=['pass_result'])

    # Map outcomes
    outcome_map = {
        'C': 'Completion',
        'Complete': 'Completion',
        'I': 'Incomplete',
        'Incomplete': 'Incomplete',
        'IN': 'Interception',
        'Interception': 'Interception'
    }
    merged['outcome'] = merged['pass_result'].map(outcome_map)
    merged = merged.dropna(subset=['outcome'])

    # Normalize time within each play
    def normalize_time(group):
        t_max = group['t'].max()
        if t_max > 0:
            group['t_norm'] = group['t'] / t_max
        else:
            group['t_norm'] = 0
        return group

    merged = merged.groupby('play_id', group_keys=False).apply(normalize_time)

    # Create time bins
    time_bins = np.linspace(0, 1, 11)
    merged['time_bin'] = pd.cut(merged['t_norm'], bins=time_bins, labels=False, include_lowest=True)

    # Calculate statistics per outcome and time bin
    summary = merged.groupby(['outcome', 'time_bin'])['dacs'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Interception': '#e74c3c',
        'Incomplete': '#3498db',
        'Completion': '#2ecc71'
    }

    markers = {
        'Interception': 'o',
        'Incomplete': 's',
        'Completion': '^'
    }

    for outcome in ['Interception', 'Incomplete', 'Completion']:
        data = summary[summary['outcome'] == outcome]
        if len(data) > 0:
            x = data['time_bin'] / 10.0
            y = data['mean']

            ax.plot(x, y, marker=markers[outcome], linewidth=2.5, label=outcome,
                   color=colors[outcome], markersize=7, alpha=0.9)

            # Add confidence interval
            if len(data[data['count'] >= 3]) > 0:
                valid = data[data['count'] >= 3]
                x_valid = valid['time_bin'] / 10.0
                y_valid = valid['mean']
                se_valid = valid['se']
                ax.fill_between(x_valid, y_valid - 1.96*se_valid, y_valid + 1.96*se_valid,
                               alpha=0.15, color=colors[outcome])

    # Add reference line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(0.02, 52, 'Equal Control', fontsize=9, color='gray', style='italic')

    # Labels and formatting
    ax.set_xlabel('Normalized Ball Flight Time (0 = Release, 1 = Arrival)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('DACS (%)', fontsize=12, fontweight='bold')
    ax.set_title('Defensive Air Control Evolution During Ball Flight',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, fontsize=11, shadow=True)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)

    # Add annotation about divergence
    ax.annotate('Control diverges\nby outcome',
                xy=(0.5, 30), xytext=(0.65, 15),
                arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5),
                fontsize=10, color='#34495e', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#34495e', alpha=0.8))

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'viz_dacs_evolution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved DACS evolution chart to {output_path}")
    plt.close()

    # Print stats
    print(f"\nStatistics:")
    print(f"Total plays: {merged['play_id'].nunique()}")
    print(f"Outcomes: {merged.groupby('outcome')['play_id'].nunique().to_dict()}")
    print(f"Avg DACS by outcome:")
    for outcome in ['Interception', 'Incomplete', 'Completion']:
        avg = merged[merged['outcome'] == outcome]['dacs'].mean()
        print(f"  {outcome}: {avg:.1f}%")

if __name__ == '__main__':
    create_evolution_chart()
