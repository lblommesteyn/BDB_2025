import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, patches
from matplotlib.lines import Line2D

if __package__ in (None, ""):
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

from analytics.dacs_one_game import (  # noqa: E402
    CORRIDOR_RADIUS_YDS,
    DT,
    analytics_input_files,
    analytics_output_files,
    load_game_output_rows,
    sample_points_in_disk,
)

FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

DEF_COLOR = (0.894, 0.102, 0.110)
WR_COLOR = (0.215, 0.494, 0.721)
FUTURE_DEF_COLOR = (1.0, 0.596, 0.0)
QB_COLOR = "#fdd835"
OFFENSE_COLOR = "#80b1d3"
GRID_COLOR = "#e6e6e6"
DEFAULT_A_LAT_MAX = 6.0

_SUPP_CACHE: Optional[pd.DataFrame] = None


def convex_hull(points: np.ndarray) -> Optional[np.ndarray]:
    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return pts if pts.size else None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def fmt(value: Optional[float], template: str = "{:.2f}") -> str:
    if value is None:
        return "--"
    try:
        if np.isnan(value):
            return "--"
    except Exception:
        pass
    return template.format(value)


def load_dacs_payload(root_dir: str, json_dir: str, game_id: int, play_id: int) -> Dict:
    json_path = os.path.join(root_dir, json_dir, f"game_{game_id}_play_{play_id}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"DACS JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_supplementary(root_dir: str) -> pd.DataFrame:
    global _SUPP_CACHE
    if _SUPP_CACHE is not None:
        return _SUPP_CACHE
    supp_path = os.path.join(
        root_dir,
        "analytics",
        "data",
        "114239_nfl_competition_files_published_analytics_final",
        "supplementary_data.csv",
    )
    if not os.path.exists(supp_path):
        _SUPP_CACHE = pd.DataFrame()
        return _SUPP_CACHE
    _SUPP_CACHE = pd.read_csv(
        supp_path,
        usecols=[
            "game_id",
            "play_id",
            "pass_result",
            "expected_points_added",
            "pass_length",
            "pass_location_type",
        ],
    )
    return _SUPP_CACHE


def load_pass_outcome(
    root_dir: str, game_id: int, play_id: int
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    supp = _load_supplementary(root_dir)
    if supp.empty:
        return None, None, None, None
    row = supp[(supp["game_id"] == game_id) & (supp["play_id"] == play_id)]
    if row.empty:
        return None, None, None, None
    r = row.iloc[0]
    return (
        None if pd.isna(r["pass_result"]) else str(r["pass_result"]),
        None if pd.isna(r["expected_points_added"]) else float(r["expected_points_added"]),
        None if pd.isna(r["pass_length"]) else str(r["pass_length"]),
        None if pd.isna(r["pass_location_type"]) else str(r["pass_location_type"]),
    )


def normalize_direction(play_direction: str, x: float, y: float) -> Tuple[float, float]:
    if str(play_direction).lower().startswith("left"):
        return 120.0 - float(x), float(y)
    return float(x), float(y)


def normalize_heading(play_direction: str, heading: float) -> float:
    if str(play_direction).lower().startswith("left"):
        return (180.0 - float(heading)) % 360.0
    return float(heading) % 360.0


def reachable_radius(t: float, s0: float, a_max: float, v_cap: float) -> float:
    s0 = max(0.0, float(s0))
    t = max(0.0, float(t))
    a_max = max(1e-6, float(a_max))
    v_cap = max(1e-6, float(v_cap))
    t_acc = max(0.0, (v_cap - s0) / a_max)
    if t <= t_acc:
        return s0 * t + 0.5 * a_max * t * t
    dist_acc = s0 * t_acc + 0.5 * a_max * t_acc * t_acc
    t_rem = t - t_acc
    return dist_acc + v_cap * t_rem


@dataclass
class PlayerSeries:
    nfl_id: int
    name: str
    position: str
    role: str
    side: str
    init_pos: np.ndarray
    speed: float
    heading_deg: float
    color: Tuple[float, float, float]
    xy: np.ndarray


def build_player_series(
    root_dir: str,
    game_id: int,
    play_id: int,
    dacs_data: Dict,
) -> Tuple[List[PlayerSeries], Optional[PlayerSeries], Optional[PlayerSeries], int, str]:
    files_in = analytics_input_files(root_dir)
    parts: List[pd.DataFrame] = []
    for fp in files_in:
        for chunk in pd.read_csv(fp, chunksize=200_000):
            sel = chunk[(chunk["game_id"] == game_id) & (chunk["play_id"] == play_id)]
            if not sel.empty:
                parts.append(sel)
    if not parts:
        raise FileNotFoundError(f"No input rows for game {game_id} play {play_id}")
    df_in = pd.concat(parts, ignore_index=True)
    frame0 = int(df_in["frame_id"].min())
    snap = df_in[df_in["frame_id"] == frame0].copy()
    play_direction = str(snap["play_direction"].iloc[0]) if "play_direction" in snap.columns else "right"

    output_files = analytics_output_files(root_dir)
    df_out = load_game_output_rows(output_files, game_id)
    df_out = df_out[df_out["play_id"] == play_id].copy()
    if df_out.empty:
        raise FileNotFoundError(f"No tracking output rows for game {game_id} play {play_id}")

    step_count = int(max(dacs_data.get("num_frames_output", 0), df_out["frame_id"].max()))
    step_count = max(step_count, 1)

    colour_map = plt.colormaps.get_cmap("tab20")

    player_series: List[PlayerSeries] = []
    wr_series: Optional[PlayerSeries] = None
    qb_series: Optional[PlayerSeries] = None

    for _, row in snap.iterrows():
        nfl_id = int(row["nfl_id"])
        x0, y0 = normalize_direction(play_direction, row["x"], row["y"])
        speed0 = float(row["s"]) if not pd.isna(row["s"]) else 0.0
        heading_deg = normalize_heading(play_direction, row["dir"]) if not pd.isna(row["dir"]) else 0.0
        color = colour_map(len(player_series) % colour_map.N)[:3]
        trace = np.full((step_count, 2), np.nan, dtype=float)
        g = df_out[df_out["nfl_id"] == nfl_id]
        for _, out_row in g.iterrows():
            k = int(out_row["frame_id"]) - 1
            if 0 <= k < step_count:
                trace[k] = normalize_direction(play_direction, out_row["x"], out_row["y"])

        ps = PlayerSeries(
            nfl_id=nfl_id,
            name=str(row.get("player_name", "")),
            position=str(row.get("player_position", "")),
            role=str(row.get("player_role", "")),
            side=str(row.get("player_side", "")),
            init_pos=np.array([x0, y0], dtype=float),
            speed=speed0,
            heading_deg=heading_deg,
            color=color,
            xy=trace,
        )

        if ps.role.lower() == "targeted receiver":
            wr_series = ps
            ps.color = WR_COLOR
        elif ps.role.lower() == "passer":
            qb_series = ps

        player_series.append(ps)

    return player_series, wr_series, qb_series, step_count, play_direction


def build_position_lookup(series: List[PlayerSeries], steps: int) -> Dict[int, np.ndarray]:
    lookup: Dict[int, np.ndarray] = {}
    for ps in series:
        arr = ps.xy.copy()
        if arr.shape[0] < steps:
            arr = np.pad(arr, ((0, steps - arr.shape[0]), (0, 0)), constant_values=np.nan)
        lookup[ps.nfl_id] = arr
    return lookup


def latest_position(arr: np.ndarray, idx: int, fallback: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return fallback
    idx = min(max(idx, 0), arr.shape[0] - 1)
    pos = arr[idx]
    if not np.any(np.isnan(pos)):
        return pos
    for j in range(idx - 1, -1, -1):
        if not np.any(np.isnan(arr[j])):
            return arr[j]
    return fallback


def compute_heading(arr: np.ndarray, idx: int, fallback_rad: float) -> float:
    if idx <= 0 or arr.size == 0:
        return fallback_rad
    prev = arr[idx - 1]
    curr = arr[idx]
    if np.any(np.isnan(prev)) or np.any(np.isnan(curr)):
        return fallback_rad
    delta = curr - prev
    if np.linalg.norm(delta) < 1e-6:
        return fallback_rad
    return math.atan2(delta[1], delta[0])


def compute_speed(arr: np.ndarray, idx: int, fallback_speed: float) -> float:
    if idx <= 0 or arr.size == 0:
        return fallback_speed
    prev = arr[idx - 1]
    curr = arr[idx]
    if np.any(np.isnan(prev)) or np.any(np.isnan(curr)):
        return fallback_speed
    dist = float(np.linalg.norm(curr - prev))
    return max(fallback_speed, dist / DT)


def draw_field(ax: plt.Axes) -> None:
    ax.set_facecolor("#0b5138")
    ax.set_xlim(FIELD_X_MIN, FIELD_X_MAX)
    ax.set_ylim(FIELD_Y_MIN, FIELD_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for x in np.arange(0, 121, 5):
        lw = 1.5 if x % 10 == 0 else 0.5
        ax.axvline(x, color=GRID_COLOR, lw=lw, alpha=0.35)
    ax.add_patch(patches.Rectangle((0, FIELD_Y_MIN), 10, FIELD_Y_MAX - FIELD_Y_MIN, facecolor="#12345b", alpha=0.55))
    ax.add_patch(patches.Rectangle((110, FIELD_Y_MIN), 10, FIELD_Y_MAX - FIELD_Y_MIN, facecolor="#12345b", alpha=0.55))


def make_animation(
    root_dir: str,
    game_id: int,
    play_id: int,
    data: Dict,
    out_path: str,
    fps: int = 12,
    dpi: int = 120,
    samples: int = 400,
    seed: int = 42,
) -> str:
    players, wr_series, qb_series, steps, play_direction = build_player_series(root_dir, game_id, play_id, data)
    positions = build_position_lookup(players, steps)
    players_by_id = {p.nfl_id: p for p in players}

    corr_centers = list(data.get("corridor_centers", []))
    if not corr_centers:
        origin = players[0].init_pos if players else np.array([60.0, 26.65])
        corr_centers = [(float(origin[0]), float(origin[1]))] * steps
    else:
        last_center = corr_centers[-1]
        while len(corr_centers) < steps:
            corr_centers.append(last_center)

    corridor_radius = float(data.get("corridor_radius", CORRIDOR_RADIUS_YDS))
    rng = np.random.default_rng(seed)
    corridor_samples = [
        sample_points_in_disk(np.array(center, dtype=float), corridor_radius, samples, rng) for center in corr_centers
    ]
    params = data.get("params", {})
    a_max = float(params.get("a_max", 4.75))
    v_cap = float(params.get("v_cap", 8.2))
    a_lat = float(params.get("a_lat_max", DEFAULT_A_LAT_MAX))

    coverage_intensity = data.get("coverage_intensity")
    bfoi = data.get("bfoi")
    dvi = data.get("dvi")
    eaepa_model = data.get("eaepa_model")
    eaepa_realized = data.get("eaepa_realized")
    event_probs = data.get("event_probabilities", {})
    pursuit_summary = data.get("pursuit_summary", {})
    hawk_summary = data.get("hawk_reaction_summary", {})
    pass_result, epa_actual, pass_length, pass_loc = load_pass_outcome(root_dir, game_id, play_id)

    share_dict = data.get("player_share_at_T", {})
    share_display: List[str] = []
    if share_dict:
        top_items = sorted(share_dict.items(), key=lambda kv: kv[1], reverse=True)[:3]
        for pid, val in top_items:
            name = players_by_id.get(pid).name if pid in players_by_id else str(pid)
            share_display.append(f"{name} {fmt(val, '{:.1f}')}%")

    defenders = [ps for ps in players if ps.side.lower() == "defense"]
    defenders.sort(key=lambda d: np.linalg.norm(d.init_pos - np.array(corr_centers[0])))
    defender_colors: Dict[int, Tuple[float, float, float]] = {d.nfl_id: DEF_COLOR for d in defenders}

    t_axis = np.arange(1, steps + 1, dtype=float) * DT

    def pad_series(arr_key: str, default: float = np.nan) -> np.ndarray:
        arr = np.asarray(data.get(arr_key, []), dtype=float)
        if arr.size == 0:
            arr = np.full(steps, default, dtype=float)
        if arr.size < steps:
            arr = np.pad(arr, (0, steps - arr.size), constant_values=default)
        else:
            arr = arr[:steps]
        return arr

    dacs_series = pad_series("dacs_series", default=0.0)
    dacs_lo = pad_series("dacs_series_lo", default=np.nan)
    dacs_hi = pad_series("dacs_series_hi", default=np.nan)
    ce_series = pad_series("coverage_entropy_norm_series", default=np.nan)
    collapse_series = pad_series("collapse_rate_series", default=0.0)

    fig = plt.figure(figsize=(10.6, 7.8), dpi=dpi)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3.6, 1.0, 0.7, 0.9], hspace=0.42)
    ax_field = fig.add_subplot(gs[0])
    ax_series = fig.add_subplot(gs[1])
    ax_delta = fig.add_subplot(gs[2], sharex=ax_series)
    ax_metrics = fig.add_subplot(gs[3])

    draw_field(ax_field)
    hud_text = fig.text(
        0.01,
        0.02,
        "",
        fontsize=10,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.55),
    )
    share_text = fig.text(
        0.99,
        0.02,
        "",
        fontsize=9,
        color="#222222",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )


    ax_series.set_facecolor("#f5f5f5")
    ax_series.fill_between(t_axis, dacs_lo, dacs_hi, color="#9ecae1", alpha=0.35, label="DACS band")
    line_dacs, = ax_series.plot(t_axis, dacs_series, color="#2171b5", lw=2.2, label="DACS%")
    ax_series.set_xlim(t_axis[0], t_axis[-1])
    ax_series.set_ylim(0, 100)
    ax_series.set_xlabel("Time since release (s)")
    ax_series.set_ylabel("DACS (%)", color="#2171b5")
    ax_series.tick_params(axis="y", labelcolor="#2171b5")

    ax_ce = ax_series.twinx()
    line_ce, = ax_ce.plot(t_axis, ce_series, color="#ff7f0e", lw=1.2, alpha=0.85, label="Coverage entropy")
    ax_ce.set_ylim(0, 1)
    ax_ce.set_ylabel("Coverage entropy", color="#ff7f0e")
    ax_ce.tick_params(axis="y", labelcolor="#ff7f0e")

    cursor_line = ax_series.axvline(t_axis[0], color="k", linestyle="--", alpha=0.6)
    cursor_point, = ax_series.plot([t_axis[0]], [dacs_series[0]], "o", color="#2171b5", ms=5)

    ax_series.legend(
        [line_dacs, line_ce, patches.Patch(facecolor="#9ecae1", alpha=0.35)],
        ["DACS%", "Coverage entropy", "DACS band"],
        loc="upper right",
        frameon=False,
    )

    delta_series = np.zeros_like(dacs_series)
    if steps > 1:
        delta_series[1:] = dacs_series[1:] - dacs_series[:-1]

    base_colors = [
        "#2ca02c" if val >= 0 else "#d62728" for val in delta_series
    ]
    bars = ax_delta.bar(
        t_axis,
        delta_series,
        width=max(0.6 * DT, 0.05),
        color=base_colors,
        alpha=0.65,
        edgecolor="none",
    )
    ax_delta.axhline(0.0, color="#444444", lw=0.8, alpha=0.6)
    ax_delta.set_ylabel("ΔDACS (pct pts)")
    ax_delta.set_xlim(t_axis[0] - DT * 0.5, t_axis[-1] + DT * 0.5)
    ax_delta.set_xlabel("Time since release (s)")
    delta_cursor_line = ax_delta.axvline(t_axis[0], color="k", linestyle="--", alpha=0.5)

    ax_metrics.axis("off")
    summary_lines = [
        f"BFOI {fmt(bfoi)} | Coverage {fmt(coverage_intensity)} | DVI {fmt(dvi)}",
        f"EAEPA {fmt(eaepa_model, '{:+.2f}')} | Realized {fmt(eaepa_realized, '{:+.2f}')} | EPA(actual) {fmt(epa_actual, '{:+.2f}')}",
        f"Pursuit mean {fmt(pursuit_summary.get('mean'))} | min {fmt(pursuit_summary.get('min'))} | Hawk mean {fmt(hawk_summary.get('mean'))}",
        f"P(C/I/INT) {fmt(event_probs.get('catch'))}/{fmt(event_probs.get('incomplete'))}/{fmt(event_probs.get('interception'))}",
    ]
    if share_display:
        summary_lines.append("Top influence: " + " | ".join(share_display))
    ax_metrics.text(
        0.01,
        0.85,
        "\n".join(summary_lines),
        transform=ax_metrics.transAxes,
        fontsize=10,
        fontweight="semibold",
        color="#333333",
        va="top",
        ha="left",
    )
    aux_lines = []
    if qb_series is not None:
        aux_lines.append(f"QB: {qb_series.name or qb_series.nfl_id}")
    if wr_series is not None:
        aux_lines.append(f"Target: {wr_series.name or wr_series.nfl_id}")
    if pass_length or pass_loc:
        aux_lines.append(f"Route: {pass_length or '--'} | {pass_loc or '--'}")
    if aux_lines:
        ax_metrics.text(
            0.99,
            0.12,
            " | ".join(aux_lines),
            transform=ax_metrics.transAxes,
            fontsize=9,
            color="#555555",
            ha="right",
            va="bottom",
        )

    coverage_legend = [
        Line2D([0], [0], color=DEF_COLOR, lw=2.5, label="Coverage now"),
        Line2D([0], [0], color=FUTURE_DEF_COLOR, lw=2.0, linestyle="--", label="Reach by catch"),
    ]
    role_legend = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=DEF_COLOR, markersize=8, label="Defense"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=WR_COLOR, markersize=8, label="Target WR"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=QB_COLOR, markersize=8, label="Quarterback"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=OFFENSE_COLOR, markersize=8, label="Other offense"),
    ]

    trail_history: Dict[int, List[np.ndarray]] = {ps.nfl_id: [] for ps in players}
    pass_outcome_displayed = False

    def animate(frame_idx: int):
        nonlocal pass_outcome_displayed
        ax_field.cla()
        draw_field(ax_field)

        center = np.array(corr_centers[min(frame_idx, len(corr_centers) - 1)], dtype=float)
        samples_now = corridor_samples[min(frame_idx, len(corridor_samples) - 1)]

        ax_field.add_patch(
            patches.Circle(
                (center[0], center[1]),
                radius=corridor_radius,
                facecolor="#a6bddb",
                alpha=0.22,
                edgecolor="#2171b5",
                linewidth=1.0,
            )
        )

        coverage_mask = np.zeros(samples_now.shape[0], dtype=bool)
        future_mask = np.zeros(samples_now.shape[0], dtype=bool)

        t_now = (frame_idx + 1) * DT
        t_future = steps * DT

        def role_color(ps: PlayerSeries) -> Tuple[float, float, float]:
            if ps is wr_series:
                return WR_COLOR
            if ps is qb_series:
                return QB_COLOR
            if (ps.side or "").lower() == "defense":
                return DEF_COLOR
            return OFFENSE_COLOR

        plotted_ids: Dict[int, np.ndarray] = {}

        for defender in defenders:
            arr = positions.get(defender.nfl_id, np.empty((0, 2)))
            center_now = latest_position(arr, frame_idx, defender.init_pos)
            heading_now = compute_heading(arr, frame_idx, math.radians(defender.heading_deg))
            speed_now = compute_speed(arr, frame_idx, defender.speed)

            cos_h = math.cos(heading_now)
            sin_h = math.sin(heading_now)

            a_axis_now = reachable_radius(t_now, speed_now, a_max, v_cap)
            b_limit_now = (max(speed_now, 1e-6) ** 2) / max(a_lat, 1e-6)
            b_axis_now = min(a_axis_now, b_limit_now)

            a_axis_future = reachable_radius(t_future, speed_now, a_max, v_cap)
            b_limit_future = (max(speed_now, 1e-6) ** 2) / max(a_lat, 1e-6)
            b_axis_future = min(a_axis_future, b_limit_future)

            dx = samples_now[:, 0] - center_now[0]
            dy = samples_now[:, 1] - center_now[1]
            xprime = dx * cos_h + dy * sin_h
            yprime = -dx * sin_h + dy * cos_h

            inside_now = (xprime / max(a_axis_now, 1e-6)) ** 2 + (yprime / max(b_axis_now, 1e-6)) ** 2 <= 1.0
            inside_future = (xprime / max(a_axis_future, 1e-6)) ** 2 + (yprime / max(b_axis_future, 1e-6)) ** 2 <= 1.0

            coverage_mask |= inside_now
            future_mask |= inside_future

            trail_history[defender.nfl_id].append(center_now.copy())
            history = np.array(trail_history[defender.nfl_id][-10:])
            if history.shape[0] > 1:
                ax_field.plot(
                    history[:, 0],
                    history[:, 1],
                    color=role_color(players_by_id[defender.nfl_id]),
                    alpha=0.5,
                    lw=1.1,
                )

            color = role_color(players_by_id[defender.nfl_id])
            ax_field.scatter(center_now[0], center_now[1], s=42, color=color, edgecolor="white", linewidths=0.6, zorder=5)
            plotted_ids[defender.nfl_id] = center_now

        uncovered_pts = samples_now[~coverage_mask]
        covered_pts = samples_now[coverage_mask]
        future_pts = samples_now[future_mask]

        if uncovered_pts.shape[0] > 0:
            ax_field.scatter(
                uncovered_pts[:, 0],
                uncovered_pts[:, 1],
                s=20,
                color="#d9d9d9",
                alpha=0.55,
                edgecolor="none",
                zorder=2,
            )

        if covered_pts.shape[0] > 0:
            ax_field.scatter(
                covered_pts[:, 0],
                covered_pts[:, 1],
                s=26,
                color=DEF_COLOR,
                alpha=0.28,
                edgecolor="none",
                zorder=3,
            )

        if future_pts.shape[0] > 0:
            ax_field.scatter(
                future_pts[:, 0],
                future_pts[:, 1],
                s=18,
                facecolor="none",
                edgecolor=FUTURE_DEF_COLOR,
                linewidths=0.8,
                alpha=0.5,
                zorder=3,
            )

        if wr_series is not None:
            wr_pos = latest_position(positions[wr_series.nfl_id], frame_idx, wr_series.init_pos)
            ax_field.scatter(
                wr_pos[0],
                wr_pos[1],
                s=68,
                color=WR_COLOR,
                edgecolor="white",
                linewidths=1.0,
                zorder=6,
            )
            plotted_ids[wr_series.nfl_id] = wr_pos

        if qb_series is not None:
            qb_pos = latest_position(
                positions[qb_series.nfl_id], min(frame_idx, positions[qb_series.nfl_id].shape[0] - 1), qb_series.init_pos
            )
            ax_field.scatter(
                qb_pos[0],
                qb_pos[1],
                s=52,
                facecolor="#f0f0f0",
                edgecolor="#444444",
                linewidths=0.9,
                zorder=5,
            )
            plotted_ids[qb_series.nfl_id] = qb_pos

        other_offense = [
            ps
            for ps in players
            if ps.nfl_id not in plotted_ids
            and ps is not wr_series
            and ps is not qb_series
            and (ps.side or "").lower() != "defense"
        ]
        for ps in other_offense:
            pos = latest_position(positions[ps.nfl_id], frame_idx, ps.init_pos)
            ax_field.scatter(
                pos[0],
                pos[1],
                s=40,
                color=OFFENSE_COLOR,
                edgecolor="white",
                linewidths=0.5,
                zorder=5,
            )
            plotted_ids[ps.nfl_id] = pos

        idx_clamped = min(frame_idx, len(dacs_series) - 1)
        dacs_now = float(dacs_series[idx_clamped])
        ce_now = float(ce_series[idx_clamped])
        collapse_now = float(collapse_series[idx_clamped])
        dacs_lo_now = float(dacs_lo[idx_clamped])
        dacs_hi_now = float(dacs_hi[idx_clamped])
        delta_now = float(delta_series[idx_clamped])

        cursor_x = t_axis[min(frame_idx, len(t_axis) - 1)]
        cursor_line.set_xdata([cursor_x] * 2)
        cursor_point.set_data([cursor_x], [dacs_now])
        delta_cursor_line.set_xdata([cursor_x] * 2)
        for idx, bar in enumerate(bars):
            bar.set_alpha(0.65)
            bar.set_edgecolor("none")
            bar.set_facecolor(base_colors[idx])
        if frame_idx < len(bars):
            bars[frame_idx].set_alpha(1.0)
            bars[frame_idx].set_edgecolor("#000000")

        hud_lines = [
            f"Frame {frame_idx + 1}/{steps}  |  t = {cursor_x:.2f}s",
            f"DACS {fmt(dacs_now, '{:.1f}')}%  (band {fmt(dacs_lo_now, '{:.1f}')}-{fmt(dacs_hi_now, '{:.1f}')}%)",
            f"Delta per frame {fmt(delta_now, '{:+.1f}')} pct pts   Delta rate {fmt(collapse_now, '{:+.1f}')}%/s   CE {fmt(ce_now)}   Coverage {fmt(coverage_intensity)}",
        ]
        hud_text.set_text("\n".join(hud_lines))
        if share_display:
            share_text.set_text("Top influence:\n" + "\n".join(share_display))
        else:
            share_text.set_text("")

        legend_cov = ax_field.legend(handles=coverage_legend, loc="upper right", frameon=False, fontsize=9)
        ax_field.add_artist(legend_cov)
        ax_field.legend(
            handles=role_legend,
            loc="upper right",
            bbox_to_anchor=(0.98, -0.08),
            ncol=4,
            frameon=False,
            fontsize=9,
        )

        if not pass_outcome_displayed and frame_idx >= steps - 1 and pass_result:
            pass_outcome_displayed = True
            label = {"C": "Complete", "I": "Incomplete", "IN": "Interception"}.get(pass_result, pass_result)
            color = {"C": "#2ca02c", "I": "#e41a1c", "IN": "#ffbf00"}.get(pass_result, "#ffffff")
            ax_field.text(
                0.99,
                0.03,
                f"Outcome: {label}",
                transform=ax_field.transAxes,
                fontsize=12,
                color=color,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.4),
            )

        return []

    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=max(35, int(1000 / max(fps, 1))), blit=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Defensive Air Control with field overlays and contextual dashboard."
    )
    parser.add_argument("--game_id", type=int, required=True)
    parser.add_argument("--play_id", type=int, required=True)
    parser.add_argument("--json_dir", type=str, default=os.path.join("analytics", "outputs", "dacs"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("analytics", "outputs", "gifs"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root_dir = os.getcwd()
    payload = load_dacs_payload(root_dir, args.json_dir, args.game_id, args.play_id)
    out_path = os.path.join(args.out_dir, f"game_{args.game_id}_play_{args.play_id}.gif")
    result_path = make_animation(
        root_dir=root_dir,
        game_id=args.game_id,
        play_id=args.play_id,
        data=payload,
        out_path=os.path.join(root_dir, out_path),
        fps=max(1, args.fps),
        dpi=max(60, args.dpi),
        samples=max(200, args.samples),
        seed=args.seed,
    )
    print(result_path)


if __name__ == "__main__":
    main()









