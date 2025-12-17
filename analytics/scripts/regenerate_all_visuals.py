"""
Regenerate key visuals using the FULL dataset
- DACS Evolution chart (with more plays)
- Eraser Leaderboard (aggregated across all games)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

def load_timeseries_data():
    """Load all timeseries data from the full dataset"""
    dacs_dir = Path('analytics/outputs/dacs_final_full')

    # Find all timeseries parquet files
    ts_files = list(dacs_dir.glob('**/game_*_dacs_timeseries.parquet'))
    print(f"Found {len(ts_files)} timeseries files")

    all_ts = []
    for i, ts_file in enumerate(ts_files):
        if i % 10 == 0:
            print(f"  Loading timeseries {i}/{len(ts_files)}...")

        df = pd.read_parquet(ts_file)
        all_ts.append(df)

    combined = pd.concat(all_ts, ignore_index=True)
    print(f"Loaded {len(combined)} timeseries rows")
    return combined

def create_evolution_chart(output_dir='analytics/outputs/presentation'):
    """Create DACS evolution chart using all available data"""

    print("\n[1/2] Creating DACS Evolution Chart...")

    # Load timeseries
    ts = load_timeseries_data()

    # Load supplementary for outcomes
    supp = pd.read_csv('analytics/data/supplementary_data.csv', low_memory=False)
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

    # Filter to meaningful plays (max DACS > 5%)
    play_max = merged.groupby('play_id')['dacs'].max()
    meaningful = play_max[play_max > 5].index
    merged = merged[merged['play_id'].isin(meaningful)]

    print(f"  Using {len(meaningful)} plays with meaningful DACS")
    print(f"  Outcomes: {merged.groupby('outcome')['play_id'].nunique().to_dict()}")

    # Normalize time
    def normalize_time(group):
        t_max = group['t'].max()
        if t_max > 0:
            group['t_norm'] = group['t'] / t_max
        else:
            group['t_norm'] = 0
        return group

    # Store play_id before groupby
    merged_with_play = merged.copy()
    merged = merged.groupby('play_id', group_keys=False).apply(lambda g: normalize_time(g))
    # Restore play_id if needed
    if 'play_id' not in merged.columns:
        merged = merged.reset_index()

    # Bin by time
    time_bins = np.linspace(0, 1, 11)
    merged['time_bin'] = pd.cut(merged['t_norm'], bins=time_bins, labels=False, include_lowest=True)

    # Calculate stats
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

            ax.plot(x, y, marker=markers[outcome], linewidth=3,
                   label=outcome, color=colors[outcome],
                   markersize=8, alpha=0.9, markeredgewidth=2,
                   markeredgecolor='white')

            # Confidence interval
            valid = data[data['count'] >= 3]
            if len(valid) > 0:
                x_v = valid['time_bin'] / 10.0
                y_v = valid['mean']
                se_v = valid['se']
                ax.fill_between(x_v,
                               np.maximum(0, y_v - 1.96*se_v),
                               y_v + 1.96*se_v,
                               alpha=0.2, color=colors[outcome])

    # Reference line
    ax.axhline(y=50, color='#7f8c8d', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(0.02, 52, 'Equal Control (50%)', fontsize=11, color='#7f8c8d',
            style='italic', weight='bold')

    # Labels
    ax.set_xlabel('Normalized Ball Flight Time\n(0 = Release, 1 = Arrival)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Defensive Air Control Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('DACS Evolution During Ball Flight by Outcome\n9,852 Plays | 2023 NFL Season',
                fontsize=15, fontweight='bold', pad=15)

    # Legend with counts
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

    plt.tight_layout()

    output_path = Path(output_dir) / 'viz_dacs_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved to {output_path}")
    plt.close()

def create_eraser_leaderboard(output_dir='analytics/outputs/presentation'):
    """Create Eraser leaderboard using all games"""

    print("\n[2/2] Creating Eraser Leaderboard...")

    dacs_dir = Path('analytics/outputs/dacs_final_full')

    # Find all summary CSVs
    summary_files = list(dacs_dir.glob('**/game_*_dacs_summary.csv'))
    print(f"  Found {len(summary_files)} game summaries")

    all_summaries = []
    for sf in summary_files:
        df = pd.read_csv(sf)
        all_summaries.append(df)

    combined = pd.concat(all_summaries, ignore_index=True)
    print(f"  Total play summaries: {len(combined)}")

    # Load player names
    players_path = Path('analytics/data/players.csv')
    if not players_path.exists():
        players_path = Path('analytics/data/raw/114239_nfl_competition_files_published_analytics_final/players.csv')

    players = pd.read_csv(players_path)
    player_map = dict(zip(players['nflId'], players['displayName']))

    # For each play, the top contributor gets credit
    # Aggregate by defender
    defender_stats = []

    for _, row in combined.iterrows():
        if pd.notna(row['top_contributor_nfl_id']):
            defender_stats.append({
                'nfl_id': int(row['top_contributor_nfl_id']),
                'dacs_contribution': row['top_contributor_ps_pct'],
                'dacs_max': row['dacs_max'],
                'play_count': 1
            })

    df_stats = pd.DataFrame(defender_stats)

    # Aggregate
    leaderboard = df_stats.groupby('nfl_id').agg({
        'dacs_contribution': 'sum',
        'dacs_max': 'mean',
        'play_count': 'sum'
    }).reset_index()

    # Filter to min plays
    leaderboard = leaderboard[leaderboard['play_count'] >= 50]

    # Sort by total contribution
    leaderboard = leaderboard.sort_values('dacs_contribution', ascending=False).head(15)

    # Map names and filter out Unknown
    leaderboard['name'] = leaderboard['nfl_id'].map(player_map)
    leaderboard = leaderboard.dropna(subset=['name'])  # Remove Unknown players

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(leaderboard)))

    bars = ax.barh(range(len(leaderboard)), leaderboard['dacs_contribution'],
                   color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for i, (val, plays) in enumerate(zip(leaderboard['dacs_contribution'], leaderboard['play_count'])):
        ax.text(val + 5, i, f'{val:.0f}  ({int(plays)} plays)',
               va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(leaderboard)))
    ax.set_yticklabels(leaderboard['name'], fontsize=12)
    ax.invert_yaxis()

    ax.set_xlabel('Total DACS Contribution (Cumulative Player Share %)', fontsize=13, fontweight='bold')
    ax.set_title('The "Erasers": Top Defenders by Air Control Contribution\n2023 NFL Season (9,852 Plays)',
                fontsize=16, fontweight='bold', pad=20, color='#2c3e50')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Add note
    fig.text(0.5, 0.02,
             'Minimum 50 plays as top contributor. Higher values = more consistent elite air control.',
             ha='center', fontsize=10, style='italic', color='#7f8c8d')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = Path(output_dir) / 'viz_eraser_leaderboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved to {output_path}")
    print(f"\n  Top 5 Erasers:")
    for i, row in leaderboard.head(5).iterrows():
        print(f"    {row['name']}: {row['dacs_contribution']:.0f} contribution ({int(row['play_count'])} plays)")
    plt.close()

if __name__ == '__main__':
    print("="*60)
    print("Regenerating Key Visuals with Full Dataset")
    print("="*60)

    create_evolution_chart()
    create_eraser_leaderboard()

    print(f"\n{'='*60}")
    print("All visuals regenerated!")
    print("="*60)
