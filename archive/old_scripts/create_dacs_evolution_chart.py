"""
Create DACS Evolution Chart showing how DACS changes during ball flight
for different outcomes (Interception, Incompletion, Completion).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def load_play_data(json_path):
    """Load DACS time series from a play JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    ts = data.get('time_series', [])
    if not ts:
        return None

    df = pd.DataFrame(ts)
    df['play_id'] = data.get('play_id')
    df['outcome'] = data.get('pass_result', 'Unknown')
    return df

def create_evolution_chart(output_dir='analytics/outputs/presentation'):
    """Create DACS evolution chart from available play data."""

    # Load all available play JSONs
    dacs_dir = Path('analytics/outputs/dacs')
    play_files = list(dacs_dir.glob('game_*_play_*.json'))

    print(f"Found {len(play_files)} play files")

    # Collect data by outcome
    all_plays = []
    for pf in play_files[:100]:  # Limit to first 100 for speed
        df = load_play_data(pf)
        if df is not None and 'dacs_pct' in df.columns:
            all_plays.append(df)

    if not all_plays:
        print("No valid play data found")
        return

    combined = pd.concat(all_plays, ignore_index=True)

    # Map outcomes to standard labels
    outcome_map = {
        'C': 'Completion',
        'Complete': 'Completion',
        'I': 'Incomplete',
        'Incomplete': 'Incomplete',
        'IN': 'Interception',
        'Interception': 'Interception'
    }
    combined['outcome_label'] = combined['outcome'].map(outcome_map)
    combined = combined.dropna(subset=['outcome_label'])

    # Normalize time to 0-1 scale (fraction of ball flight)
    def normalize_time(group):
        t = group['time_since_throw'].values
        if len(t) > 0 and t.max() > 0:
            group['time_norm'] = t / t.max()
        else:
            group['time_norm'] = 0
        return group

    combined = combined.groupby('play_id', group_keys=False).apply(normalize_time)

    # Create bins for normalized time
    time_bins = np.linspace(0, 1, 11)  # 10 bins
    combined['time_bin'] = pd.cut(combined['time_norm'], bins=time_bins, labels=False, include_lowest=True)

    # Calculate mean DACS per time bin per outcome
    summary = combined.groupby(['outcome_label', 'time_bin'])['dacs_pct'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Interception': '#e74c3c',
        'Incomplete': '#3498db',
        'Completion': '#2ecc71'
    }

    for outcome in ['Interception', 'Incomplete', 'Completion']:
        data = summary[summary['outcome_label'] == outcome]
        if len(data) > 0:
            x = data['time_bin'] / 10.0  # Convert back to 0-1
            y = data['mean']

            ax.plot(x, y, marker='o', linewidth=2.5, label=outcome,
                   color=colors.get(outcome, '#95a5a6'), markersize=6)

            # Add confidence band if enough data
            if 'se' in data.columns:
                ax.fill_between(x, y - 1.96*data['se'], y + 1.96*data['se'],
                               alpha=0.2, color=colors.get(outcome, '#95a5a6'))

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.5, 52, 'Equal Control', ha='center', fontsize=9, color='gray')

    ax.set_xlabel('Normalized Ball Flight Time (0 = Release, 1 = Arrival)', fontsize=12, fontweight='bold')
    ax.set_ylabel('DACS (%)', fontsize=12, fontweight='bold')
    ax.set_title('DACS Evolution During Ball Flight by Outcome', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)

    # Add annotation
    ax.annotate('Defenders converge\non interceptions',
                xy=(0.6, 75), xytext=(0.7, 85),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                fontsize=10, color='#e74c3c', fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir) / 'viz_dacs_evolution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved DACS evolution chart to {output_path}")
    plt.close()

    # Print summary stats
    print("\nSummary Statistics:")
    print(f"Total plays analyzed: {combined['play_id'].nunique()}")
    print(f"Outcomes: {combined['outcome_label'].value_counts().to_dict()}")

if __name__ == '__main__':
    create_evolution_chart()
