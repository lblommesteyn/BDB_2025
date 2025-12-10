import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
else:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from analytics.outcome_model import load_outcome_model, predict_event_probs, FEATURE_COLUMNS

def load_season_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Season summary not found at {path}")
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding='utf-8', encoding_errors='replace')

def plot_calibration_curve(y_true, y_prob, title, ax):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=title)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_title(f"Calibration: {title}")
    ax.legend()

def plot_roc_curve(y_true, y_prob, title, ax):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC: {title}')
    ax.legend(loc="lower right")

def generate_hardest_routes(df: pd.DataFrame, out_dir: str):
    # Ensure route info is present (might need to merge with supplementary if not in summary)
    # Assuming summary has 'route_of_targeted_receiver' or similar if it was in supplementary row
    # If not, we might need to load supplementary data again. 
    # For now, check if column exists.
    
    route_col = 'route_of_targeted_receiver'
    if route_col not in df.columns:
        print(f"Warning: {route_col} not in dataframe. Skipping hardest routes.")
        return

    # Filter for reasonable sample size
    counts = df[route_col].value_counts()
    valid_routes = counts[counts >= 10].index
    df_filtered = df[df[route_col].isin(valid_routes)]

    grouped = df_filtered.groupby(route_col).agg(
        dacs_final=('dacs_final', 'mean'),
        expected_points_added=('expected_points_added', 'mean'),
        count=('dacs_final', 'count')
    ).sort_values('dacs_final')

    out_path = os.path.join(out_dir, 'hardest_routes.csv')
    grouped.to_csv(out_path)
    print(f"Saved hardest routes to {out_path}")

def generate_player_rankings(df: pd.DataFrame, out_dir: str):
    # This requires player-level data. The summary might be play-level.
    # If summary is play-level, we can't do player rankings easily unless we have player share columns exploded.
    # batch_runner produces 'season_play_metrics.parquet'.
    # Let's check if we can use top_contributors.csv if it exists, or if we need to parse the JSONs.
    # For this script, let's assume we might need to rely on what we have or skip if data missing.
    pass

def generate_model_plots(df: pd.DataFrame, model_path: str, out_dir: str):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Skipping model plots.")
        return

    try:
        bundle = load_outcome_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Prepare features
    # We need to map df columns to FEATURE_COLUMNS
    # Assuming df has these columns from batch_runner output
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feats:
        print(f"Missing features for model: {missing_feats}. Skipping model plots.")
        return

    X = df[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    
    # Predict
    # bundle.clf is LogisticRegression
    # We need scaled features
    X_scaled = bundle.scaler.transform(X)
    probs = bundle.clf.predict_proba(X_scaled)
    
    # We need ground truth. 'pass_result' column?
    if 'pass_result' not in df.columns:
        print("Missing 'pass_result' column. Skipping evaluation plots.")
        return

    y_true = df['pass_result']
    # Map to classes
    # classes_: ['catch', 'incomplete', 'interception']
    # y_true might be 'C', 'I', 'IN'? or 'Complete', 'Incomplete', 'Interception'?
    # Let's standardize
    y_map = {'C': 'catch', 'I': 'incomplete', 'IN': 'interception', 
             'Complete': 'catch', 'Incomplete': 'incomplete', 'Interception': 'interception'}
    y_mapped = y_true.map(y_map)
    
    # Filter rows where result is known
    mask = y_mapped.isin(bundle.classes_)
    if not mask.any():
        print("No valid labels found for evaluation.")
        return
        
    X_eval = X_scaled[mask]
    y_eval = y_mapped[mask]
    probs_eval = bundle.clf.predict_proba(X_eval)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC for 'catch' (class 0 usually, check bundle.classes_)
    # bundle.classes_ is likely a numpy array
    if isinstance(bundle.classes_, np.ndarray):
        catch_idx = np.where(bundle.classes_ == 'catch')[0][0]
    else:
        catch_idx = bundle.classes_.index('catch')
    y_binary = (y_eval == 'catch').astype(int)
    plot_roc_curve(y_binary, probs_eval[:, catch_idx], 'Catch Probability', axes[0])
    
    # Calibration for 'catch'
    plot_calibration_curve(y_binary, probs_eval[:, catch_idx], 'Catch Probability', axes[1])
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'model_evaluation_plots.png')
    plt.savefig(out_path)
    print(f"Saved model plots to {out_path}")

def generate_eraser_leaderboard(df: pd.DataFrame, out_dir: str):
    # "Eraser" metric: Total EPA Prevented
    # We define EPA Prevented as: (Baseline EPA for the pass) - (Actual EPA)
    # But DACS is about preventing completions.
    # A better metric might be: Sum of (Expected EPA of Completion * (1 - DACS_final_prob)) - Actual EPA?
    # Let's stick to the definition in the plan: 
    # "Ranking defenders based on their total EPA prevented."
    # We can use 'expected_epa_coverage' vs 'actual_epa'.
    # expected_epa_coverage is derived from DACS probabilities.
    # EPA Prevented = expected_epa_coverage - actual_epa (if positive, defender did better than expected)
    # Wait, if expected_epa_coverage is low (good defense) and actual is low (incomplete), difference is small.
    # If expected is high (bad defense) and actual is low (incomplete), difference is large (big play prevented).
    
    if 'expected_epa_coverage' not in df.columns or 'actual_epa' not in df.columns:
        print("Missing EPA columns. Skipping Eraser leaderboard.")
        return

    # Filter for plays with a top contributor
    if 'top_contributor_nfl_id' not in df.columns:
        print("Missing top_contributor_nfl_id. Skipping Eraser leaderboard.")
        return

    df['epa_prevented'] = df['expected_epa_coverage'] - df['actual_epa']
    
    # Group by defender
    # We need defender names. 'top_contributor_nfl_id' is just ID.
    # We might not have names here. We can try to load players.csv or just use IDs for now.
    # Or maybe 'top_contributor_name' is in the summary? No, looking at batch_runner it saves what dacs_one_game returns.
    # dacs_one_game returns 'top_ps_id' but not name in the summary df construction (need to check dacs_one_game.py).
    # Let's assume ID for now.
    
    leaderboard = df.groupby('top_contributor_nfl_id').agg(
        plays=('play_id', 'count'),
        total_epa_prevented=('epa_prevented', 'sum'),
        avg_dacs=('dacs_final', 'mean'),
        avg_collapse=('peak_collapse_rate', 'mean')
    ).sort_values('total_epa_prevented', ascending=False)
    
    # Filter for minimum plays
    leaderboard = leaderboard[leaderboard['plays'] >= 20]
    
    out_path = os.path.join(out_dir, 'eraser_leaderboard.csv')
    leaderboard.to_csv(out_path)
    print(f"Saved Eraser leaderboard to {out_path}")

def generate_scheme_comparison(df: pd.DataFrame, out_dir: str):
    scheme_col = 'team_coverage_man_zone'
    
    # If missing, try to merge with supplementary data
    if scheme_col not in df.columns:
        print(f"{scheme_col} missing. Attempting to merge with supplementary data...")
        try:
            # We assume REPO_ROOT is available or we can deduce it
            # REPO_ROOT is defined at top of script
            supp_path = os.path.join(REPO_ROOT, 'analytics', 'data', 'raw', '114239_nfl_competition_files_published_analytics_final', 'supplementary_data.csv')
            if os.path.exists(supp_path):
                supp = pd.read_csv(supp_path, usecols=['game_id', 'play_id', scheme_col])
                # Merge
                df = df.merge(supp, on=['game_id', 'play_id'], how='left')
                print(f"Merged {scheme_col} from supplementary data.")
            else:
                print(f"Supplementary data not found at {supp_path}. Skipping scheme comparison.")
                return
        except Exception as e:
            print(f"Failed to merge supplementary data: {e}. Skipping scheme comparison.")
            return

    if scheme_col not in df.columns:
        print(f"Still missing {scheme_col}. Skipping scheme comparison.")
        return

    grouped = df.groupby(scheme_col).agg(
        dacs_final=('dacs_final', 'mean'),
        epa_prevented=('epa_prevented', 'mean') if 'epa_prevented' in df.columns else ('dacs_final', 'count'),
        count=('dacs_final', 'count')
    )
    
    out_path = os.path.join(out_dir, 'scheme_comparison.csv')
    grouped.to_csv(out_path)
    print(f"Saved scheme comparison to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, help='Path to season summary parquet/csv')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--model', help='Path to outcome model joblib')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    try:
        df = load_season_summary(args.summary)
        print(f"Loaded {len(df)} rows from summary.")
    except Exception as e:
        print(f"Error loading summary: {e}")
        return

    generate_hardest_routes(df, args.out)
    generate_eraser_leaderboard(df, args.out)
    generate_scheme_comparison(df, args.out)
    
    if args.model:
        generate_model_plots(df, args.model, args.out)

if __name__ == '__main__':
    main()
