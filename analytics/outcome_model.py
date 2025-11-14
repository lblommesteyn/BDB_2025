import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import StandardScaler

OUTCOME_CLASSES = ["catch", "incomplete", "interception"]
TARGET_COLUMN = "actual_event_id"
FEATURE_COLUMNS = [
    "n_defenders",
    "corridor_length",
    "num_frames_output",
    "dacs_final",
    "dacs_final_lo",
    "dacs_final_hi",
    "coverage_intensity",
    "dvi",
    "bfoi",
    "prob_catch",
    "prob_incomplete",
    "prob_interception",
]


@dataclass
class OutcomeModelBundle:
    scaler: StandardScaler
    clf: LogisticRegression
    features: Sequence[str]
    classes_: Sequence[str]


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _split_by_fold(
    df: pd.DataFrame, val_fold: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folds = df.get("fold_id")
    if folds is not None:
        mask = folds.to_numpy() == val_fold
    else:
        # Random 20% validation if folds missing
        mask = np.zeros(len(df), dtype=bool)
        mask[: int(0.2 * len(df))] = True
        rng = np.random.default_rng(42)
        rng.shuffle(mask)
    X = df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return X[~mask], X[mask], y[~mask], y[mask]


def train_outcome_model(
    df: pd.DataFrame,
    max_iter: int = 1000,
    val_fold: int = 0,
) -> Tuple[OutcomeModelBundle, Dict[str, float], Dict[str, Dict[str, float]]]:
    X_train, X_val, y_train, y_val = _split_by_fold(df, val_fold=val_fold)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=max_iter,
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)
    train_probs = clf.predict_proba(X_train_s)
    val_probs = clf.predict_proba(X_val_s)
    labels = list(range(len(OUTCOME_CLASSES)))
    metrics = {
        "train_log_loss": float(log_loss(y_train, train_probs)),
        "val_log_loss": float(log_loss(y_val, val_probs, labels=labels)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }
    reports = {
        "train_report": classification_report(
            y_train,
            train_probs.argmax(axis=1),
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "val_report": classification_report(
            y_val,
            val_probs.argmax(axis=1),
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    bundle = OutcomeModelBundle(
        scaler=scaler,
        clf=clf,
        features=list(FEATURE_COLUMNS),
        classes_=list(OUTCOME_CLASSES),
    )
    return bundle, metrics, reports


def save_model(bundle: OutcomeModelBundle, path: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "scaler": bundle.scaler,
        "clf": bundle.clf,
        "features": bundle.features,
        "classes": bundle.classes_,
        "metrics": metrics,
    }
    joblib.dump(payload, path)


def load_outcome_model(path: str) -> OutcomeModelBundle:
    payload = joblib.load(path)
    return OutcomeModelBundle(
        scaler=payload["scaler"],
        clf=payload["clf"],
        features=payload["features"],
        classes_=payload["classes"],
    )


def predict_event_probs(bundle: OutcomeModelBundle, features: np.ndarray) -> Dict[str, float]:
    X = bundle.scaler.transform(features.reshape(1, -1))
    probs = bundle.clf.predict_proba(X)[0]
    return {cls: float(prob) for cls, prob in zip(bundle.classes_, probs)}


def feature_vector_from_play(result: Dict, heuristics: Dict[str, float]) -> np.ndarray:
    values = [
        result.get("n_defenders", 0),
        result.get("corridor_length", 0.0),
        result.get("num_frames_output", 0),
        result.get("dacs_final", 0.0),
        result.get("dacs_final_lo", 0.0),
        result.get("dacs_final_hi", 0.0),
        result.get("coverage_intensity", 0.0),
        result.get("dvi", 0.0),
        result.get("bfoi", 0.0),
        heuristics.get("catch", np.nan),
        heuristics.get("incomplete", np.nan),
        heuristics.get("interception", np.nan),
    ]
    arr = np.array([0 if v is None else v for v in values], dtype=np.float32)
    arr[np.isnan(arr)] = 0.0
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the outcome probability model.")
    parser.add_argument(
        "--data",
        default=os.path.join("analytics", "data", "outcome_training.parquet"),
        help="Path to the prepared training dataset.",
    )
    parser.add_argument(
        "--dest",
        default=os.path.join("analytics", "models", "outcome_model.joblib"),
        help="Destination for the trained model bundle.",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=0,
        help="Fold id to use for validation (requires fold_id column).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for the logistic regression solver.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(args.data)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    bundle, metrics, reports = train_outcome_model(
        df, max_iter=args.max_iter, val_fold=args.val_fold
    )
    save_model(bundle, args.dest, metrics)
    print(f"[outcome_model] Saved model to {args.dest}")
    print("[outcome_model] Metrics:", metrics)
    print("[outcome_model] Validation classification report:")
    # Simple textual summary
    for label, stats in reports["val_report"].items():
        if not isinstance(stats, dict):
            continue
        if label in ("accuracy",):
            print(f"  {label}: {stats}")
        else:
            print(
                f"  {label}: precision={stats.get('precision'):.3f}, "
                f"recall={stats.get('recall'):.3f}, f1={stats.get('f1-score'):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
