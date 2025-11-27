import pandas as pd
import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season-metrics", required=True)
    parser.add_argument("--supplementary", required=True)
    parser.add_argument("--out-parquet", required=True)
    args = parser.parse_args()

    print(f"Reading metrics from {args.season_metrics}...")
    df = pd.read_parquet(args.season_metrics)
    
    print(f"Reading supplementary data from {args.supplementary}...")
    supp_cols = [
        "game_id", "play_id", "pass_result", "expected_points_added",
        "pass_length", "pass_location_type", "route_of_targeted_receiver",
        "team_coverage_type", "team_coverage_man_zone", "dropback_type",
        "dropback_distance"
    ]
    supp = pd.read_csv(args.supplementary, usecols=supp_cols)
    
    print("Merging...")
    merged = df.merge(supp, on=["game_id", "play_id"], how="left", suffixes=("", "_supp"))
    
    # Calculate derived columns
    merged["epa_diff_model_vs_actual"] = merged["expected_epa_coverage"] - merged["actual_epa"]
    merged["epa_diff_model_vs_supp"] = merged["expected_epa_coverage"] - merged["expected_points_added"]
    
    # Rename columns to match outcome_model expectations
    rename_map = {
        "peak_collapse_rate": "collapse_rate_peak",
        "top_contributor_ps_pct": "ps_norm_top1",
        "prob_catch": "prob_catch_prior",
        "prob_incomplete": "prob_incomplete_prior",
        "prob_interception": "prob_interception_prior"
    }
    merged.rename(columns=rename_map, inplace=True)

    # Fill missing features expected by model
    if "collapse_rate_mean" not in merged.columns:
        merged["collapse_rate_mean"] = 0.0 # Not in summary
    if "ps_norm_top2" not in merged.columns:
        merged["ps_norm_top2"] = 0.0 # Not in summary

    if "dacs_final_p95" in merged.columns and "dacs_final_p05" in merged.columns:
        merged["uncertainty_width"] = merged["dacs_final_p95"] - merged["dacs_final_p05"]
    else:
        merged["uncertainty_width"] = np.nan
        
    print(f"Saving {len(merged)} rows to {args.out_parquet}...")
    merged.to_parquet(args.out_parquet)
    print("Done.")

if __name__ == "__main__":
    main()
