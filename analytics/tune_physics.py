import os
import glob
import json
import argparse
import pandas as pd
import numpy as np

def get_input_files(root_dir):
    base = os.path.join(root_dir, 'analytics', 'data', 'raw', '114239_nfl_competition_files_published_analytics_final', 'train')
    files = sorted(glob.glob(os.path.join(base, 'input_2023_w*.csv')))
    return files

def analyze_physics(root_dir, sample_files=5):
    files = get_input_files(root_dir)
    if not files:
        print("No input files found.")
        return

    # Use a subset of files to save time
    files = files[:sample_files]
    print(f"Analyzing {len(files)} files for physics tuning...")

    all_speeds = []
    all_accels = []

    for f in files:
        print(f"Reading {f}...")
        try:
            # Read only necessary columns
            df = pd.read_csv(f, usecols=['s', 'a', 'play_direction'], engine='python')
            
            # Filter out invalid values if any (though usually pre-cleaned)
            s = df['s'].dropna().values
            a = df['a'].dropna().values
            
            # Subsample to avoid massive memory usage if needed, but for 5 files it's fine
            all_speeds.append(s)
            all_accels.append(a)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_speeds:
        print("No data collected.")
        return

    cat_speeds = np.concatenate(all_speeds)
    cat_accels = np.concatenate(all_accels)

    # Calculate quantiles
    # Speed: p99 as v_cap
    v_cap = float(np.percentile(cat_speeds, 99))
    # Accel: p95 as a_max (p99 might be outliers/noise)
    a_max = float(np.percentile(cat_accels, 95))

    print(f"Calculated Parameters:")
    print(f"  v_cap (p99 speed): {v_cap:.2f} yds/s")
    print(f"  a_max (p95 accel): {a_max:.2f} yds/s^2")

    # Structure for eda_summary.json
    output = {
        "analytics_input": {
            "speed_quantiles": {
                "p99": v_cap,
                "p100": float(np.max(cat_speeds))
            },
            "accel_quantiles": {
                "p95": a_max,
                "p99": float(np.percentile(cat_accels, 99)),
                "p100": float(np.max(cat_accels))
            }
        }
    }

    out_path = os.path.join(root_dir, 'eda_summary.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved calibration to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    parser.add_argument('--samples', type=int, default=5, help='Number of files to sample')
    args = parser.parse_args()

    analyze_physics(args.root, args.samples)
