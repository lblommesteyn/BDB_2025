import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "dacs_final",
    "dacs_final_lo",
    "dacs_final_hi",
    "coverage_intensity",
    "collapse_rate_mean",
    "collapse_rate_peak",
    "ps_norm_top1",
    "ps_norm_top2",
    "ps_norm_mid_top1",
    "air_control_war_top1",
    "uncertainty_width",
    "prob_catch_prior",
    "prob_incomplete_prior",
    "prob_interception_prior",
]

CAT_COLS = [
    "route_of_targeted_receiver",
    "team_coverage_type",
    "team_coverage_man_zone",
    "pass_length",
    "pass_location_type",
    "dropback_type",
]
TOPK_PER_CAT = 15


@dataclass
class OutcomeModelBundle:
    features: List[str]
    scaler: Optional[StandardScaler]
    classifier: object
    incomplete_share: float
    feature_order: List[str]
    cat_levels: Dict[str, List[str]]
    model_type: str


def _encode_categoricals(
    df: pd.DataFrame, fit: bool, cat_levels: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    levels_out = cat_levels.copy()
    dummies: List[pd.Series] = []
    for col in CAT_COLS:
        series = df[col].fillna("UNK").astype(str) if col in df else pd.Series("UNK", index=df.index)
        if fit:
            top = series.value_counts().nlargest(TOPK_PER_CAT).index.tolist()
            if "OTHER" not in top:
                top.append("OTHER")
            levels_out[col] = top
        levels = levels_out.get(col, [])
        mapped = series.where(series.isin(levels), "OTHER" if "OTHER" in levels else "UNK")
        dummy = pd.get_dummies(mapped)
        for lev in levels:
            name = f"{col}__{lev}"
            dummies.append(dummy.get(lev, pd.Series(0, index=df.index)).rename(name))
    if dummies:
        cat_df = pd.concat(dummies, axis=1)
    else:
        cat_df = pd.DataFrame(index=df.index)
    return cat_df, levels_out


def _build_design_matrix(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    fit_cat_levels: bool,
    cat_levels: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    num_df = df[list(numeric_features)].fillna(0.0)
    cat_df, levels = _encode_categoricals(df, fit=fit_cat_levels, cat_levels=cat_levels)
    X = pd.concat([num_df, cat_df], axis=1)
    return X, levels


def _payload_to_frame(payload: Dict) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in payload.items()}])


def build_feature_vector_from_payload(
    bundle: OutcomeModelBundle,
    payload: Dict,
    heuristic_probs: Dict[str, float],
) -> np.ndarray:
    row: Dict[str, float] = {}
    for name in bundle.features:
        if name in payload and payload[name] is not None:
            row[name] = float(payload[name])
        elif name == "prob_catch_prior":
            row[name] = float(heuristic_probs.get("catch", 0.0))
        elif name == "prob_incomplete_prior":
            row[name] = float(heuristic_probs.get("incomplete", 0.0))
        elif name == "prob_interception_prior":
            row[name] = float(heuristic_probs.get("interception", 0.0))
        else:
            row[name] = 0.0
    # attach categoricals
    for col in CAT_COLS:
        row[col] = payload.get(col, "UNK")
    df = _payload_to_frame(row)
    X_num = df[bundle.features].fillna(0.0)
    cat_cols = []
    for col, levels in bundle.cat_levels.items():
        series = df[col].fillna("UNK").astype(str) if col in df else pd.Series("UNK")
        mapped = series.where(series.isin(levels), "OTHER" if "OTHER" in levels else "UNK")
        dummy = pd.get_dummies(mapped)
        for lev in levels:
            name = f"{col}__{lev}"
            cat_cols.append(dummy.get(lev, pd.Series(0)).rename(name))
    if cat_cols:
        cat_df = pd.concat(cat_cols, axis=1)
        X_full = pd.concat([X_num, cat_df], axis=1)
    else:
        X_full = X_num
    # ensure column order
    for col in bundle.feature_order:
        if col not in X_full:
            X_full[col] = 0.0
    X_full = X_full[bundle.feature_order]
    arr = X_full.to_numpy(dtype=float)
    if bundle.scaler is not None:
        arr = bundle.scaler.transform(arr)
    return arr


def predict_event_probs(
    bundle: OutcomeModelBundle, payload: Dict, heuristic_probs: Dict[str, float]
) -> Dict[str, float]:
    feats = build_feature_vector_from_payload(bundle, payload, heuristic_probs)
    prob_complete = float(bundle.classifier.predict_proba(feats)[0, 1])
    prob_complete = float(np.clip(prob_complete, 0.0, 1.0))
    remainder = max(0.0, 1.0 - prob_complete)
    prob_incomplete = remainder * bundle.incomplete_share
    prob_interception = remainder - prob_incomplete
    return {
        "catch": prob_complete,
        "incomplete": prob_incomplete,
        "interception": prob_interception,
    }


def load_outcome_model(path: str) -> Optional[OutcomeModelBundle]:
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, OutcomeModelBundle):
            return bundle
        # backward compatibility for older checkpoints
        if isinstance(bundle, dict) and "classifier" in bundle:
            return OutcomeModelBundle(
                features=bundle.get("features", []),
                scaler=bundle.get("scaler"),
                classifier=bundle.get("classifier"),
                incomplete_share=bundle.get("incomplete_share", 1.0),
                feature_order=bundle.get("feature_order", bundle.get("features", [])),
                cat_levels=bundle.get("cat_levels", {}),
                model_type=bundle.get("model_type", "logreg"),
            )
    except Exception:
        return None
    return None


def train_model_from_dataset(
    dataset_path: str,
    feature_names: Sequence[str],
    val_split: float = 0.2,
    random_state: int = 42,
    use_class_balance: bool = False,
    model_type: str = "histgb",
) -> Tuple[OutcomeModelBundle, Dict[str, float]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.lower().endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    df = df[df["pass_result"].isin(["C", "I", "IN"])].copy()
    if df.empty:
        raise ValueError("Dataset contains no labeled plays for training.")

    y = (df["pass_result"] == "C").astype(int)
    if y.nunique() < 2:
        raise ValueError("Need at least one completion and one non-completion to train.")

    numeric_features = list(feature_names)
    X_full, cat_levels = _build_design_matrix(
        df, numeric_features=numeric_features, fit_cat_levels=True, cat_levels={}
    )

    X = X_full
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=max(0.0, min(0.9, val_split)),
        stratify=y,
        random_state=random_state,
    )
    feature_order = list(X_train.columns)

    if model_type.lower() == "logreg":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if use_class_balance else None,
            solver="lbfgs",
        )
        clf.fit(X_train_s, y_train)
        prob_train = clf.predict_proba(X_train_s)[:, 1]
        prob_val = clf.predict_proba(X_val_s)[:, 1] if len(X_val_s) else np.array([])
    else:
        scaler = None
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            random_state=random_state,
            class_weight="balanced" if use_class_balance else None,
        )
        clf.fit(X_train.to_numpy(dtype=float), y_train)
        prob_train = clf.predict_proba(X_train.to_numpy(dtype=float))[:, 1]
        prob_val = (
            clf.predict_proba(X_val.to_numpy(dtype=float))[:, 1] if len(X_val) else np.array([])
        )
    metrics: Dict[str, float] = {}
    try:
        metrics["train_auc"] = float(roc_auc_score(y_train, prob_train))
    except Exception:
        metrics["train_auc"] = float("nan")
    try:
        metrics["val_auc"] = float(roc_auc_score(y_val, prob_val)) if len(prob_val) else float("nan")
    except Exception:
        metrics["val_auc"] = float("nan")
    metrics["train_brier"] = float(brier_score_loss(y_train, prob_train))
    metrics["val_brier"] = (
        float(brier_score_loss(y_val, prob_val)) if len(prob_val) else float("nan")
    )

    # Baseline using heuristic priors
    if "prob_catch_prior" in df:
        base_probs = df["prob_catch_prior"].fillna(0.0)
        try:
            metrics["baseline_auc"] = float(roc_auc_score(y, base_probs))
        except Exception:
            metrics["baseline_auc"] = float("nan")
        metrics["baseline_brier"] = float(brier_score_loss(y, base_probs))

    totals = df["pass_result"].value_counts()
    inc = float(totals.get("I", 0.0))
    inte = float(totals.get("IN", 0.0))
    denom = inc + inte
    incomplete_share = inc / denom if denom > 0 else 1.0

    return (
        OutcomeModelBundle(
            features=list(numeric_features),
            scaler=scaler,
            classifier=clf,
            incomplete_share=float(np.clip(incomplete_share, 0.0, 1.0)),
            feature_order=feature_order,
            cat_levels=cat_levels,
            model_type=model_type.lower(),
        ),
        {**metrics, "n": int(len(df))},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an outcome calibration model from DACS evaluation data."
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(
            "analytics", "data", "dacs_eval", "dacs_eval_wk1_3.parquet"
        ),
        help="Path to harvested evaluation dataset (Parquet or CSV).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("analytics", "models", "outcome_model.joblib"),
        help="Output path for the trained model bundle.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction for reporting metrics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    parser.add_argument(
        "--class-balance",
        action="store_true",
        help="Use class_weight='balanced' in logistic regression.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logreg", "histgb"],
        default="histgb",
        help="Model family to train (logistic regression or histogram gradient boosting).",
    )
    parser.add_argument(
        "--feature",
        action="append",
        dest="features",
        default=None,
        help="Feature to include (can specify multiple). Defaults to a standard list.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_names = args.features or DEFAULT_FEATURES
    bundle, metrics = train_model_from_dataset(
        args.dataset,
        feature_names,
        val_split=args.val_split,
        random_state=args.random_state,
        use_class_balance=args.class_balance,
        model_type=args.model_type,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(
        f"[outcome_model] Trained logistic calibration on {len(bundle.features)} features. "
        f"Saved to {args.out}"
    )
    print(
        f"[outcome_model] Incomplete share among non-completions: {bundle.incomplete_share:.3f}"
    )
    if metrics:
        print(
            "[outcome_model] "
            f"Train AUC: {metrics.get('train_auc', float('nan')):.3f} | "
            f"Val AUC: {metrics.get('val_auc', float('nan')):.3f} | "
            f"Train Brier: {metrics.get('train_brier', float('nan')):.3f} | "
            f"Val Brier: {metrics.get('val_brier', float('nan')):.3f} | "
            f"Baseline AUC: {metrics.get('baseline_auc', float('nan')):.3f} | "
            f"Baseline Brier: {metrics.get('baseline_brier', float('nan')):.3f} | "
            f"N={metrics.get('n', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
