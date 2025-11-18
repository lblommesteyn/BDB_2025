import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd

if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
else:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from analytics.dacs_one_game import compute_dacs_for_game, list_games_quick
from analytics.visualize_dacs import make_animation

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analytics", "outputs", "dacs")
DEFAULT_BATCH_DIR = os.path.join(REPO_ROOT, "analytics", "outputs", "batch_runner")
DEFAULT_MANIFEST = os.path.join(DEFAULT_BATCH_DIR, "manifest.jsonl")
DEFAULT_SUMMARY = os.path.join(DEFAULT_BATCH_DIR, "season_play_metrics.parquet")
DEFAULT_GIF_DIR = os.path.join(REPO_ROOT, "analytics", "outputs", "gifs")
UTC = datetime.utcnow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DACS analytics pipeline across many games."
    )
    parser.add_argument(
        "--root",
        default=REPO_ROOT,
        help="Repository root containing analytics/ and prediction/ folders.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help="Base output directory for per-game artifacts.",
    )
    parser.add_argument(
        "--games",
        default="all",
        help="Comma-separated list of game_ids, path to a text file, or 'all'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of games to process after filtering.",
    )
    parser.add_argument(
        "--catalog-limit",
        type=int,
        default=10000,
        help="Max rows to load from supplementary data when listing games.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Manifest JSONL path capturing per-game status.",
    )
    parser.add_argument(
        "--season-summary",
        default=DEFAULT_SUMMARY,
        help="Destination Parquet file for concatenated per-play metrics.",
    )
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        help="Append to manifest instead of overwriting.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip games that already have a summary CSV in the target folder.",
    )
    parser.add_argument(
        "--samples-per-t",
        type=int,
        default=None,
        help="Override samples per timestep for compute_dacs_for_game.",
    )
    parser.add_argument(
        "--corridor-radius",
        type=float,
        default=None,
        help="Override ball corridor sampling radius (yards).",
    )
    parser.add_argument(
        "--topk-ps",
        type=int,
        default=None,
        help="Override number of defenders evaluated for Player Share.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed forwarded to compute_dacs_for_game.",
    )
    parser.add_argument(
        "--residual-model-path",
        default=None,
        help="Path to a residual model checkpoint to override the default.",
    )
    parser.add_argument(
        "--no-residual-model",
        action="store_true",
        help="Disable residual reach corrections.",
    )
    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=0,
        help="Number of Monte Carlo samples for uncertainty propagation (0 disables).",
    )
    parser.add_argument(
        "--outcome-model-path",
        default=None,
        help="Path to a calibrated outcome model checkpoint.",
    )
    parser.add_argument(
        "--no-outcome-model",
        action="store_true",
        help="Disable outcome model calibration (fallback to heuristics).",
    )
    parser.add_argument(
        "--make-gifs",
        action="store_true",
        help="After each game, render GIFs for every successfully processed play.",
    )
    parser.add_argument(
        "--gif-dir",
        default=DEFAULT_GIF_DIR,
        help="Destination directory for GIF outputs when --make-gifs is set.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frame rate used for GIF rendering.",
    )
    parser.add_argument(
        "--gif-dpi",
        type=int,
        default=120,
        help="Resolution used for GIF rendering.",
    )
    parser.add_argument(
        "--gif-samples",
        type=int,
        default=400,
        help="Number of samples per frame for GIF visualization.",
    )
    parser.add_argument(
        "--gif-skip-existing",
        action="store_true",
        help="Skip GIF generation when the output file already exists.",
    )
    return parser.parse_args()


def _read_games_arg(arg: str) -> List[int]:
    """Parse --games argument into a list of integers."""
    arg = arg.strip()
    if not arg or arg.lower() == "all":
        return []
    if os.path.exists(arg):
        values: List[int] = []
        with open(arg, "r", encoding="utf-8") as f:
            for line in f:
                val = line.strip()
                if not val:
                    continue
                values.append(int(val))
        return values
    return [int(token.strip()) for token in arg.split(",") if token.strip()]


def _filter_game_ids(
    catalog: pd.DataFrame, requested: Iterable[int], limit: Optional[int]
) -> List[int]:
    """Intersect requested ids with catalog and enforce ordering + limit."""
    available = catalog["game_id"].astype(int).tolist()
    if not requested:
        ids = available
    else:
        lookup = set(requested)
        ids = [gid for gid in available if gid in lookup]
        missing = sorted(lookup - set(ids))
        if missing:
            print(f"[batch_runner] Warning: {len(missing)} requested game_ids missing: {missing[:5]}")
    if limit is not None:
        ids = ids[:limit]
    return ids


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _summary_already_exists(game_out_dir: str, game_id: int) -> bool:
    csv_path = os.path.join(game_out_dir, f"game_{game_id}_dacs_summary.csv")
    return os.path.exists(csv_path)


def _gif_output_path(gif_dir: str, game_id: int, play_id: int) -> str:
    return os.path.join(gif_dir, f"game_{game_id}_play_{play_id}.gif")


def _render_gifs(
    *,
    root_dir: str,
    json_paths: Iterable[str],
    gif_dir: str,
    fps: int,
    dpi: int,
    samples: int,
    seed: Optional[int],
    skip_existing: bool,
) -> List[str]:
    rendered: List[str] = []
    os.makedirs(gif_dir, exist_ok=True)
    for json_path in json_paths:
        fname = os.path.basename(json_path)
        if not fname.startswith("game_") or "_play_" not in fname:
            print(f"[batch_runner] Unable to parse game/play ids from {fname}; skipping GIF.")
            continue
        try:
            game_part, play_part = fname.split("_play_")
            game_id = int(game_part.replace("game_", "").strip())
            play_id = int(play_part.replace(".json", "").strip())
        except ValueError:
            print(f"[batch_runner] Unable to parse game/play ids from {fname}; skipping GIF.")
            continue

        gif_path = _gif_output_path(gif_dir, game_id, play_id)
        if skip_existing and os.path.exists(gif_path):
            rendered.append(gif_path)
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            make_animation(
                root_dir=root_dir,
                game_id=game_id,
                play_id=play_id,
                data=payload,
                out_path=gif_path,
                fps=max(1, fps),
                dpi=max(60, dpi),
                samples=max(50, samples),
                seed=seed,
            )
            rendered.append(gif_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[batch_runner] GIF generation failed for {fname}: {exc}")
    return rendered


def _write_manifest(entries: List[dict], path: str, append: bool) -> None:
    mode = "a" if append and os.path.exists(path) else "w"
    _ensure_dir(os.path.dirname(path))
    with open(path, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry))
            f.write("\n")


def _write_season_summary(df: pd.DataFrame, dest: str) -> str:
    _ensure_dir(os.path.dirname(dest))
    try:
        df.to_parquet(dest, index=False)
        return dest
    except Exception as err:
        print(f"[batch_runner] Parquet write failed ({err}); falling back to CSV.")
        csv_path = os.path.splitext(dest)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> int:
    args = parse_args()
    args.root = os.path.abspath(args.root)

    def resolve_path(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(args.root, path)

    args.out = resolve_path(args.out)
    args.manifest = resolve_path(args.manifest)
    args.season_summary = resolve_path(args.season_summary)
    args.gif_dir = resolve_path(args.gif_dir)

    catalog = list_games_quick(args.root, limit=args.catalog_limit)
    requested = _read_games_arg(args.games)
    game_ids = _filter_game_ids(catalog, requested, args.limit)
    if not game_ids:
        print("[batch_runner] No games resolved from inputs. Nothing to do.")
        return 0

    manifest_entries: List[dict] = []
    season_rows: List[pd.DataFrame] = []

    print(
        f"[batch_runner] Starting run for {len(game_ids)} games. Outputs -> {args.out}"
    )
    for idx, game_id in enumerate(game_ids, start=1):
        game_out_dir = os.path.join(args.out, f"game_{game_id}")
        _ensure_dir(game_out_dir)
        if args.skip_complete and _summary_already_exists(game_out_dir, game_id):
            entry = {
                "game_id": game_id,
                "status": "skipped",
                "reason": "exists",
                "output_dir": game_out_dir,
                "timestamp_utc": UTC().isoformat() + "Z",
            }
            manifest_entries.append(entry)
            print(f"[batch_runner] [{idx}/{len(game_ids)}] Skip game {game_id} (already complete).")
            continue

        print(f"[batch_runner] [{idx}/{len(game_ids)}] Processing game {game_id}...")
        start_time = time.perf_counter()
        compute_kwargs = {}
        if args.samples_per_t is not None:
            compute_kwargs["samples_per_t"] = args.samples_per_t
        if args.corridor_radius is not None:
            compute_kwargs["corridor_radius"] = args.corridor_radius
        if args.topk_ps is not None:
            compute_kwargs["topk_ps"] = args.topk_ps
        if args.seed is not None:
            compute_kwargs["seed"] = args.seed
        if args.residual_model_path is not None:
            compute_kwargs["residual_model_path"] = args.residual_model_path
        if args.no_residual_model:
            compute_kwargs["use_residual_model"] = False
        if args.uncertainty_samples and args.uncertainty_samples > 0:
            compute_kwargs["uncertainty_samples"] = args.uncertainty_samples
        if args.outcome_model_path is not None:
            compute_kwargs["outcome_model_path"] = args.outcome_model_path
        if args.no_outcome_model:
            compute_kwargs["use_outcome_model"] = False

        status = "success"
        error_msg = None
        per_play_json_count = 0
        play_count = 0
        gif_paths: List[str] = []
        try:
            df_summary, json_paths = compute_dacs_for_game(
                root_dir=args.root,
                game_id=game_id,
                out_dir=game_out_dir,
                **compute_kwargs,
            )
            play_count = len(df_summary.index)
            per_play_json_count = len(json_paths)
            season_rows.append(df_summary)
            if args.make_gifs and json_paths:
                gif_paths = _render_gifs(
                    root_dir=args.root,
                    json_paths=json_paths,
                    gif_dir=args.gif_dir,
                    fps=args.gif_fps,
                    dpi=args.gif_dpi,
                    samples=args.gif_samples,
                    seed=args.seed,
                    skip_existing=args.gif_skip_existing,
                )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_msg = repr(exc)
            print(f"[batch_runner] Game {game_id} failed: {exc}")

        duration = time.perf_counter() - start_time
        entry = {
            "game_id": game_id,
            "status": status,
            "play_count": play_count,
            "per_play_json_count": per_play_json_count,
            "gif_count": len(gif_paths),
            "duration_sec": round(duration, 3),
            "output_dir": game_out_dir,
            "timestamp_utc": UTC().isoformat() + "Z",
        }
        if error_msg:
            entry["error"] = error_msg
        manifest_entries.append(entry)

    if manifest_entries:
        _write_manifest(manifest_entries, args.manifest, args.append_manifest)
        print(f"[batch_runner] Manifest written to {args.manifest}")

    if season_rows:
        season_df = pd.concat(season_rows, ignore_index=True)
        summary_path = _write_season_summary(season_df, args.season_summary)
        print(f"[batch_runner] Season summary saved to {summary_path}")
    else:
        print("[batch_runner] No successful games; season summary not written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
