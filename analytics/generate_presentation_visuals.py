import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import math

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

def load_data(summary_path):
    if summary_path.endswith('.parquet'):
        return pd.read_parquet(summary_path)
    return pd.read_csv(summary_path)

def plot_eraser_leaderboard(df, out_dir):
    # Top 10 Defenders by Total EPA Prevented
    # We need to calculate EPA prevented if not present
    if 'epa_prevented' not in df.columns:
        if 'expected_epa_coverage' in df.columns and 'actual_epa' in df.columns:
            df['epa_prevented'] = df['expected_epa_coverage'] - df['actual_epa']
        else:
            print("Cannot calculate EPA prevented. Skipping leaderboard.")
            return

    # Group by defender ID
    # We don't have names in summary, so we'll use ID for now. 
    # In a real scenario, we'd merge with players.csv.
    # Let's try to load players.csv if available.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    players_path = os.path.join(repo_root, "analytics", "data", "raw", "114239_nfl_competition_files_published_analytics_final", "players.csv")
    
    player_map = {}
    if os.path.exists(players_path):
        try:
            pdf = pd.read_csv(players_path)
            player_map = dict(zip(pdf['nflId'], pdf['displayName']))
        except:
            pass

    leaderboard = df.groupby('top_contributor_nfl_id').agg(
        total_epa_prevented=('epa_prevented', 'sum'),
        plays=('play_id', 'count')
    ).sort_values('total_epa_prevented', ascending=False).head(10)
    
    leaderboard = leaderboard[leaderboard['plays'] >= 20] # Minimum sample
    
    # Map names
    def get_name(x):
        try:
            if pd.isna(x): return "Unknown"
            return player_map.get(int(x), str(int(x)))
        except:
            return str(x)

    leaderboard['name'] = leaderboard.index.map(get_name)
    
    plt.figure(figsize=(12, 6))
    if leaderboard.empty:
        print("Leaderboard empty. Skipping plot.")
        return

    ax = sns.barplot(x='total_epa_prevented', y='name', data=leaderboard, palette='viridis')
    plt.title('The Erasers: Top Defenders by EPA Prevented', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Total EPA Prevented', fontsize=12)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'viz_eraser_leaderboard.png'), dpi=300)
    plt.close()
    print("Generated Eraser Leaderboard.")

def plot_shutdown_curve(df, out_dir):
    # DACS over time for different outcomes
    # We need time-series data. The summary only has scalars.
    # BUT, batch_runner saves 'dacs_timeseries.parquet' for each game.
    # Loading ALL time series is heavy.
    # Maybe we can just use the summary stats? No, user wants a curve.
    # We can't easily do this without the full time series data.
    # Alternative: Use 'time_to_50pct' distribution?
    # Or, load a sample of games (e.g. 20 random games) to approximate the curve.
    
    print("Generating Shutdown Curve (sampling 20 games)...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dacs_dir = os.path.join(repo_root, "analytics", "outputs", "dacs_final_full")
    
    # Get list of game dirs
    game_dirs = [d for d in os.listdir(dacs_dir) if d.startswith('game_')]
    sample_games = np.random.choice(game_dirs, size=min(len(game_dirs), 20), replace=False)
    
    all_series = []
    
    for gdir in sample_games:
        ts_path = os.path.join(dacs_dir, gdir, f"{gdir}_dacs_timeseries.parquet")
        if os.path.exists(ts_path):
            try:
                ts_df = pd.read_parquet(ts_path)
                # Ensure play_id is int
                ts_df['play_id'] = ts_df['play_id'].astype(int)
                
                # Filter summary for this game
                gid = int(gdir.split('_')[1])
                game_summary = df[df['game_id'] == gid][['play_id', 'pass_result']].copy()
                game_summary['play_id'] = game_summary['play_id'].astype(int)
                
                merged = ts_df.merge(game_summary, on='play_id', how='inner')
                if not merged.empty:
                    all_series.append(merged)
            except Exception as e:
                print(f"Error processing {gdir}: {e}")
                
    if not all_series:
        print("No time series data found. Skipping Shutdown Curve.")
        return

    full_ts = pd.concat(all_series, ignore_index=True)
    print(f"Shutdown Curve data: {len(full_ts)} rows.")
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='time_sec', y='dacs', hue='pass_result', data=full_ts, ci=95)
    plt.title('The Shutdown Curve: DACS Evolution by Outcome', fontsize=16, fontweight='bold')
    plt.xlabel('Time Since Snap (s)')
    plt.ylabel('DACS (%)')
    plt.xlim(0, 4)
    plt.ylim(0, 100)
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'viz_shutdown_curve.png'), dpi=300)
    plt.close()
    print("Generated Shutdown Curve.")

def plot_scheme_radar(df, out_dir):
    # Man vs Zone
    if 'team_coverage_man_zone' not in df.columns:
        # Try merge
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        supp_path = os.path.join(repo_root, 'analytics', 'data', 'raw', '114239_nfl_competition_files_published_analytics_final', 'supplementary_data.csv')
        if os.path.exists(supp_path):
            supp = pd.read_csv(supp_path, usecols=['game_id', 'play_id', 'team_coverage_man_zone'])
            df = df.merge(supp, on=['game_id', 'play_id'], how='left')
            
    if 'team_coverage_man_zone' not in df.columns:
        print("No scheme data. Skipping Radar.")
        return

    # Metrics to compare
    # Normalize them to 0-1 for radar
    # 1. DACS Final (Higher is better)
    # 2. EPA Prevented (Higher is better)
    # 3. Completion % (Lower is better -> 1 - Comp%)
    # 4. Collapse Rate (Higher is better)
    
    # Calculate Comp %
    df['is_complete'] = (df['pass_result'] == 'C').astype(int)
    
    grouped = df.groupby('team_coverage_man_zone').agg(
        avg_dacs=('dacs_final', 'mean'),
        avg_epa_prev=('epa_prevented', 'mean'),
        comp_pct=('is_complete', 'mean'),
        avg_collapse=('peak_collapse_rate', 'mean')
    )
    
    # Invert comp pct
    grouped['inv_comp_pct'] = 1 - grouped['comp_pct']
    
    # Select only Man and Zone
    target_schemes = ['Man', 'Zone']
    grouped = grouped[grouped.index.isin(target_schemes)]
    
    if grouped.empty:
        print("No Man/Zone data.")
        return

    # Normalize columns
    for col in ['avg_dacs', 'avg_epa_prev', 'inv_comp_pct', 'avg_collapse']:
        grouped[col + '_norm'] = (grouped[col] - grouped[col].min()) / (grouped[col].max() - grouped[col].min() + 1e-6)
        # Actually radar charts usually show raw values on different axes or normalized to a common scale.
        # Let's just plot raw values on separate axes? No, radar needs common scale usually.
        # Let's just use 0-1 normalization relative to the max observed in the dataset?
        # Or just plot two lines.
        pass

    # Let's use a simple bar chart comparison instead of radar if radar is too complex to implement without plotly
    # Radar in matplotlib is tricky.
    # Let's do a grouped bar chart.
    
    metrics = ['avg_dacs', 'avg_epa_prev', 'inv_comp_pct', 'avg_collapse']
    labels = ['Avg DACS', 'EPA Prevented', 'Incompletion %', 'Collapse Rate']
    
    # Normalize for visualization
    plot_data = grouped[metrics].copy()
    for col in metrics:
        plot_data[col] = plot_data[col] / plot_data[col].max()
        
    plot_data = plot_data.reset_index().melt(id_vars='team_coverage_man_zone', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='team_coverage_man_zone', data=plot_data)
    plt.title('Scheme Comparison: Man vs. Zone (Normalized)', fontsize=16, fontweight='bold')
    plt.xticks(range(4), labels)
    plt.ylabel('Relative Score (Higher is Better)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'viz_scheme_comparison.png'), dpi=300)
    plt.close()
    print("Generated Scheme Comparison.")

def plot_route_heatmap(df, out_dir):
    # Pass Location x Pass Length
    if 'pass_location_type' not in df.columns or 'pass_length' not in df.columns:
        print("Missing pass location info. Skipping Heatmap.")
        return
        
    pivot = df.pivot_table(index='pass_length', columns='pass_location_type', values='dacs_final', aggfunc='mean')
    
    # Reorder
    length_order = ['short', 'deep']
    loc_order = ['left', 'middle', 'right']
    
    pivot = pivot.reindex(index=length_order, columns=loc_order)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=100)
    plt.title('Route Difficulty: Avg DACS by Target Area', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'viz_route_heatmap.png'), dpi=300)
    plt.close()
    print("Generated Route Heatmap.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    df = load_data(args.summary)
    print(f"Loaded {len(df)} rows.")
    
    try:
        plot_eraser_leaderboard(df, args.out)
    except Exception as e:
        print(f"Failed to plot Eraser Leaderboard: {e}")

    try:
        plot_shutdown_curve(df, args.out)
    except Exception as e:
        print(f"Failed to plot Shutdown Curve: {e}")

    try:
        plot_scheme_radar(df, args.out)
    except Exception as e:
        print(f"Failed to plot Scheme Radar: {e}")

    try:
        plot_route_heatmap(df, args.out)
    except Exception as e:
        print(f"Failed to plot Route Heatmap: {e}")

if __name__ == "__main__":
    main()
