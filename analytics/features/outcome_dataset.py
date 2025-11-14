"""
Utilities for assembling an outcome calibration dataset from DACS play outputs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ANALYTICS_BASE = os.path.join(
    "analytics", "data", "114239_nfl_competition_files_published_analytics_final"
)
DEFAULT_OUTPUTS = os.path.join("analytics", "outputs", "dacs")
DEFAULT_OUTFILE = os.path.join("analytics", "data", "outcome_training.parquet")
PASS_RESULT_MAP = {"C": "catch", "I": "incomplete", "IN": "interception"}
OUTCOME_ORDER = ["catch", "incomplete", "interception"]

SUPP_USECOLS = [
    "game_id",
    "play_id",
    "pass_result",
    "pass_length",
    "pass_location_type",
    "route_of_targeted_receiver",
    "dropback_type",
    "dropback_distance",
    "team_coverage_type",
    "team_coverage_man_zone",
    "expected_points_added",
]

SUPP_DTYPES = {
    "game_id": "int64",
    "play_id": "int64",
    "pass_result": "category",
    "pass_length": "float32",
    "pass_location_type": "category",
    "route_of_targeted_receiver": "category",
    "dropback_type": "category",
    "dropback_distance": "float32",
    "team_coverage_type": "category",
    "team_coverage_man_zone": "category",
    "expected_points_added": "float32",
}


@dataclass(frozen=True)
class PlayKey:
    game_id: int
    play_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile per-play coverage outcome training data."
    )
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Repository root containing analytics/outputs and analytics/data.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=DEFAULT_OUTPUTS,
        help="Directory containing per-play JSON outputs (recursively searched).",
    )
    parser.add_argument(
        "--supplementary",
        default=os.path.join(ANALYTICS_BASE, "supplementary_data.csv"),
        help="Path to supplementary_data.csv with ground-truth pass results.",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_OUTFILE,
        help="Destination Parquet path for the merged dataset.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of deterministic folds (fold 0 is treated as validation).",
    )
    parser.add_argument(
        "--glob-pattern",
        default="game_*_play_*.json",
        help="Glob pattern (relative) to locate per-play JSON files.",
    )
    return parser.parse_args()


def _make_outputs_glob(outputs_dir: str, pattern: str) -> str:
    if any(p in pattern for p in ("*", "?")):
        return os.path.join(outputs_dir, "**", pattern)
    return os.path.join(outputs_dir, "**", "game_*_play_*.json")


def _load_supplementary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"supplementary data not found: {path}")
    df = pd.read_csv(path, usecols=SUPP_USECOLS, dtype=SUPP_DTYPES)
    df["pass_result"] = df["pass_result"].astype(str).str.upper()
    return df


def _supplementary_lookup(df: pd.DataFrame) -> Dict[PlayKey, pd.Series]:
    df = df.set_index(["game_id", "play_id"])
    lookup: Dict[PlayKey, pd.Series] = {}
    for (game_id, play_id), row in df.iterrows():
        lookup[PlayKey(int(game_id), int(play_id))] = row
    return lookup


def _iter_play_jsons(glob_expr: str) -> Iterable[str]:
    for path in glob.iglob(glob_expr, recursive=True):
        if os.path.isfile(path):
            yield path


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_records(outputs_dir: str, pattern: str) -> List[Dict]:
    glob_expr = _make_outputs_glob(outputs_dir, pattern)
    records: List[Dict] = []
    for path in _iter_play_jsons(glob_expr):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        event_probs = data.get("event_probabilities_prior") or data.get("event_probabilities")
        if not event_probs:
            continue
        rec = {
            "game_id": int(data.get("game_id")),
            "play_id": int(data.get("play_id")),
            "n_defenders": int(data.get("n_defenders", 0) or 0),
            "corridor_length": _safe_float(data.get("corridor_length")),
            "num_frames_output": int(data.get("num_frames_output", 0) or 0),
            "dacs_final": _safe_float(data.get("dacs_final")),
            "dacs_final_lo": _safe_float(data.get("dacs_final_lo")),
            "dacs_final_hi": _safe_float(data.get("dacs_final_hi")),
            "coverage_intensity": _safe_float(data.get("coverage_intensity")),
            "dvi": _safe_float(data.get("dvi")),
            "bfoi": _safe_float(data.get("bfoi")),
            "prob_catch": _safe_float(event_probs.get("catch")),
            "prob_incomplete": _safe_float(event_probs.get("incomplete")),
            "prob_interception": _safe_float(event_probs.get("interception")),
        }
        records.append(rec)
    return records


def merge_with_labels(records: List[Dict], supp_lookup: Dict[PlayKey, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    keys = [PlayKey(row.game_id, row.play_id) for row in df.itertuples()]
    supp_rows: List[pd.Series] = []
    for key in keys:
        row = supp_lookup.get(key)
        supp_rows.append(row)
    supp_df = pd.DataFrame(supp_rows)
    merged = pd.concat([df.reset_index(drop=True), supp_df.reset_index(drop=True)], axis=1)
    merged["actual_event"] = merged["pass_result"].map(lambda v: PASS_RESULT_MAP.get(str(v).upper()))
    outcome_idx = {name: idx for idx, name in enumerate(OUTCOME_ORDER)}
    merged["actual_event_id"] = merged["actual_event"].map(outcome_idx)
    return merged


def assign_folds(df: pd.DataFrame, num_folds: int) -> pd.DataFrame:
    if df.empty:
        return df
    seeds = df["game_id"].astype(np.int64) * 1315423911 + df["play_id"].astype(np.int64)
    folds = (seeds % num_folds).astype(int)
    df = df.copy()
    df["fold_id"] = folds
    df["split"] = np.where(df["fold_id"] == 0, "val", "train")
    return df


def main() -> int:
    args = parse_args()
    outputs_dir = args.outputs_dir
    if not os.path.isabs(outputs_dir):
        outputs_dir = os.path.join(args.root, outputs_dir)
    supp_path = args.supplementary
    if not os.path.isabs(supp_path):
        supp_path = os.path.join(args.root, supp_path)
    dest = args.dest
    if not os.path.isabs(dest):
        dest = os.path.join(args.root, dest)

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    records = collect_records(outputs_dir, args.glob_pattern)
    if not records:
        print("[outcome_dataset] No per-play JSON records found.")
        return 1

    supp_df = _load_supplementary(supp_path)
    lookup = _supplementary_lookup(supp_df)
    merged = merge_with_labels(records, lookup)
    merged = assign_folds(merged, args.num_folds)

    merged.to_parquet(dest, index=False)
    print(f"[outcome_dataset] Saved dataset with {len(merged)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
