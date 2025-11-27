# %% [markdown]
# # DACS Advanced Analysis
# This notebook generates the "Eraser" leaderboard and performs situational analysis (Coverage Schemes).

# %%
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Run DACS advanced analysis.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(REPO_ROOT, "analytics", "data", "dacs_eval", "dacs_eval_full.parquet"),
        help="Path to the full evaluation dataset."
    )
    parser.add_argument(
        "--players",
        default=os.path.join(REPO_ROOT, "analytics", "data", "114239_nfl_competition_files_published_analytics_final", "players.csv"),
        help="Path to players.csv for name lookup."
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "analytics", "outputs"),
        help="Directory to save plots."
    )
    
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # %%
    # Load Data
    print(f"Loading dataset from {args.dataset}...")
    df = pd.read_parquet(args.dataset)
    
    # Load Players for names
    players_map = {}
    players_pos = {}
    if os.path.exists(args.players):
        print(f"Loading players from {args.players}...")
        players_df = pd.read_csv(args.players)
        players_map = players_df.set_index("nflId")["displayName"].to_dict()
        players_pos = players_df.set_index("nflId")["position"].to_dict()
    else:
        print(f"[WARN] players.csv not found at {args.players}. Using IDs.")

    # %%
    # 1. The "Eraser" Leaderboard
    # We use 'top_contributor_nfl_id' and 'eaepa_realized' (or expected_epa_coverage)
    # We want to sum the EPA prevented by each player.
    # Note: 'top_contributor_nfl_id' is the player with the highest share on that play.
    # This is a simplification but a good proxy for the "primary defender".
    
    # Filter for valid plays
    valid_plays = df.dropna(subset=["top_contributor_nfl_id", "eaepa_realized"])
    
    # Group by defender
    leaderboard = valid_plays.groupby("top_contributor_nfl_id").agg(
        plays=("play_id", "count"),
        total_epa_prevented=("eaepa_realized", "sum"),
        avg_dacs=("dacs_final", "mean"),
        avg_collapse=("collapse_rate_peak", "mean")
    ).reset_index()
    
    # Filter for minimum plays to be significant
    print(f"Found {len(leaderboard)} defenders with at least one contribution.")
    leaderboard = leaderboard[leaderboard["plays"] >= 5].copy()
    print(f"Retained {len(leaderboard)} defenders with >= 5 contributions.")
    
    # Map names
    leaderboard["player_name"] = leaderboard["top_contributor_nfl_id"].map(players_map).fillna("Unknown")
    leaderboard["position"] = leaderboard["top_contributor_nfl_id"].map(players_pos).fillna("UNK")
    
    # Sort by Total EPA Prevented (Descending)
    leaderboard = leaderboard.sort_values("total_epa_prevented", ascending=False)
    
    print("\nTop 10 'Erasers' (Total EPA Prevented):")
    print(leaderboard[["player_name", "position", "plays", "total_epa_prevented", "avg_dacs"]].head(10))
    
    # Save leaderboard
    leaderboard.to_csv(os.path.join(args.out_dir, "eraser_leaderboard.csv"), index=False)
    
    # Plot Top 20
    plt.figure(figsize=(12, 8))
    top20 = leaderboard.head(20)
    # Check if we have valid positions for hue
    if top20["position"].nunique() > 1:
        sns.barplot(data=top20, x="total_epa_prevented", y="player_name", hue="position", dodge=False)
    else:
        sns.barplot(data=top20, x="total_epa_prevented", y="player_name")
    plt.title("Top 20 Defenders by Total Air EPA Prevented (The 'Erasers')")
    plt.xlabel("Total Expected Points Saved")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "eraser_leaderboard.png"))
    # plt.show()

    # %%
    # 2. Situational Analysis: Scheme Wars
    # Compare DACS metrics by Coverage Type
    
    # Clean up coverage types
    # 'team_coverage_type' often has many specific types. Let's group them.
    # Common: 'Cover 3', 'Cover 1', 'Cover 4', 'Cover 2', 'Cover 6', '2 Man'
    
    if "team_coverage_type" in df.columns:
        cov_stats = df.groupby("team_coverage_type").agg(
            plays=("play_id", "count"),
            avg_dacs=("dacs_final", "mean"),
            avg_collapse=("collapse_rate_peak", "mean"),
            avg_epa_prevented=("eaepa_realized", "mean")
        ).reset_index()
        
        # Filter rare coverages
        cov_stats = cov_stats[cov_stats["plays"] > 50].sort_values("avg_epa_prevented", ascending=False)
        
        print("\nCoverage Scheme Performance:")
        print(cov_stats)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=cov_stats, x="avg_dacs", y="avg_collapse", size="plays", hue="team_coverage_type", sizes=(100, 1000))
        plt.title("Coverage Scheme Profile: Control vs Collapse")
        plt.xlabel("Average DACS% (Space Denied)")
        plt.ylabel("Peak Collapse Rate (Closing Speed)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "scheme_wars.png"))
        # plt.show()

    # %%
    # 3. Route Analysis
    # Which routes are hardest to cover?
    if "route_of_targeted_receiver" in df.columns:
        route_stats = df.groupby("route_of_targeted_receiver").agg(
            plays=("play_id", "count"),
            avg_dacs=("dacs_final", "mean"),
            avg_epa_prevented=("eaepa_realized", "mean")
        ).reset_index()
        
        route_stats = route_stats[route_stats["plays"] > 50].sort_values("avg_dacs", ascending=True)
        
        print("\nRoute Difficulty (Lowest DACS = Hardest to Cover):")
        print(route_stats.head(10))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=route_stats.head(15), x="avg_dacs", y="route_of_targeted_receiver", palette="viridis")
        plt.title("Hardest Routes to Cover (Lowest Average DACS%)")
        plt.xlabel("Average DACS%")
        plt.ylabel("Route")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "route_difficulty.png"))
        # plt.show()

if __name__ == "__main__":
    main()
