# %% [markdown]
# # DACS Validation Report
# This notebook validates the Defensive Air Control (DACS) metrics by analyzing their ability to predict pass outcomes.
# We compare the DACS-enhanced model against a baseline model.

# %%
import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# Add repo root to path
if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)

from analytics.outcome_model import OutcomeModelBundle, build_feature_vector_from_payload

def main():
    parser = argparse.ArgumentParser(description="Generate DACS validation report.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(REPO_ROOT, "analytics", "data", "dacs_eval", "dacs_eval_pilot.parquet"),
        help="Path to the evaluation dataset (Parquet)."
    )
    parser.add_argument(
        "--model",
        default=os.path.join(REPO_ROOT, "analytics", "models", "outcome_model_pilot.joblib"),
        help="Path to the trained outcome model."
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "analytics", "outputs"),
        help="Directory to save plots."
    )
    
    # Check if running in interactive mode (Jupyter) or script
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        args = parser.parse_args([]) # Use defaults in notebook
    else:
        args = parser.parse_args()

    # %%
    # Load Data
    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}")
        sys.exit(1)

    df = pd.read_parquet(args.dataset)
    print(f"Loaded {len(df)} plays from {args.dataset}")

    # Filter for relevant outcomes
    df = df[df["pass_result"].isin(["C", "I", "IN"])].copy()
    df["is_complete"] = (df["pass_result"] == "C").astype(int)
    print(f"Filtered to {len(df)} plays (C/I/IN).")

    # %%
    # Load Model
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        sys.exit(1)

    bundle = joblib.load(args.model)
    print(f"Loaded model type: {bundle.model_type} from {args.model}")
    print(f"Features: {bundle.features}")

    # %%
    # Generate Predictions
    X_list = []
    for _, row in df.iterrows():
        heuristics = {
            "catch": row.get("prob_catch_prior", 0),
            "incomplete": row.get("prob_incomplete_prior", 0),
            "interception": row.get("prob_interception_prior", 0),
        }
        vec = build_feature_vector_from_payload(bundle, row.to_dict(), heuristics)
        X_list.append(vec)

    X = np.vstack(X_list)
    y_true = df["is_complete"].values

    # Predict
    y_prob = bundle.classifier.predict_proba(X)[:, 1]
    df["prob_complete_model"] = y_prob

    # %%
    # Baseline Predictions (Heuristic)
    if "prob_catch_prior" in df.columns:
        y_prob_base = df["prob_catch_prior"].fillna(0.5).values
    else:
        y_prob_base = np.zeros_like(y_true) + 0.5

    # %%
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fpr_base, tpr_base, _ = roc_curve(y_true, y_prob_base)
    base_auc = auc(fpr_base, tpr_base)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DACS Model (AUC = {roc_auc:.2f})')
    plt.plot(fpr_base, tpr_base, color='navy', lw=2, linestyle='--', label=f'Baseline (AUC = {base_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    out_roc = os.path.join(args.out_dir, "roc_curve.png")
    plt.savefig(out_roc)
    print(f"Saved ROC curve to {out_roc}")
    # plt.show()

    # %%
    # Plot 2: Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    prob_true_base, prob_pred_base = calibration_curve(y_true, y_prob_base, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='DACS Model')
    plt.plot(prob_pred_base, prob_true_base, marker='s', linestyle='--', label='Baseline')
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot (Reliability Diagram)')
    plt.legend()
    plt.grid(alpha=0.3)
    out_cal = os.path.join(args.out_dir, "calibration_curve.png")
    plt.savefig(out_cal)
    print(f"Saved calibration curve to {out_cal}")
    # plt.show()

    # %%
    # Plot 3: Feature Importance
    if hasattr(bundle.classifier, "feature_importances_"):
        importances = bundle.classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = np.array(bundle.feature_order)
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(10), importances[indices][:10], align="center")
        plt.yticks(range(10), feature_names[indices][:10])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        out_feat = os.path.join(args.out_dir, "feature_importance.png")
        plt.savefig(out_feat)
        print(f"Saved feature importance to {out_feat}")
        # plt.show()
    else:
        print("Model does not expose feature_importances_")

if __name__ == "__main__":
    main()
