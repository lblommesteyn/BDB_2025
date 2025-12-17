import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, patches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

if __package__ in (None, ""):
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

from analytics.dacs_one_game import (
    CORRIDOR_RADIUS_YDS,
    DT,
    analytics_input_files,
    analytics_output_files,
    load_game_output_rows,
    sample_points_in_disk,
)

# -----------------------------------------------------------------------------
# Constants & Config
# -----------------------------------------------------------------------------
FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

# Color Palette (Professional & Distinct)
COLOR_DEFENSE = "#d32f2f"       # Red
COLOR_OFFENSE = "#1976d2"       # Blue
COLOR_QB = "#fbc02d"            # Yellow/Gold
COLOR_WR = "#0288d1"            # Light Blue
COLOR_FIELD = "#2e7d32"         # Green
COLOR_FIELD_ACCENT = "#388e3c"  # Lighter Green
COLOR_GRID = "#ffffff"          # White
COLOR_DACS_LINE = "#1565c0"     # Dark Blue
COLOR_DACS_BAND = "#90caf9"     # Light Blue
COLOR_CE_LINE = "#ef6c00"       # Orange
COLOR_FUTURE_REACH = "#ffca28"  # Amber

DEFAULT_A_LAT_MAX = 6.0

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
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
    color: str
    xy: np.ndarray

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
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

def latest_position(arr: np.ndarray, idx: int, fallback: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return fallback
    idx = min(max(idx, 0), arr.shape[0] - 1)
    pos = arr[idx]
    if not np.any(np.isnan(pos)):
        return pos
    # Backtrack for last valid
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

def fmt(value: Optional[float], template: str = "{:.2f}") -> str:
    if value is None: return "--"
    try:
        if np.isnan(value): return "--"
    except: pass
    return template.format(value)

# -----------------------------------------------------------------------------
# Visualizer Class
# -----------------------------------------------------------------------------
class DACSVisualizer:
    def __init__(self, root_dir: str, game_id: int, play_id: int, data: Dict, fps: int = 12, dpi: int = 120, samples: int = 400, seed: int = 42):
        self.root_dir = root_dir
        self.game_id = game_id
        self.play_id = play_id
        self.data = data
        self.fps = fps
        self.dpi = dpi
        self.samples = samples
        self.seed = seed
        
        # Load Data
        self._load_data()
        self._setup_physics()
        self._setup_corridor()
        
    def _load_data(self):
        # Load tracking data
        self.players, self.wr_series, self.qb_series, self.steps, self.play_direction = self._build_player_series()
        self.positions = self._build_position_lookup()
        self.players_by_id = {p.nfl_id: p for p in self.players}
        self.defenders = [p for p in self.players if (p.side or "").lower() == "defense"]
        
        # Load metrics series
        self.dacs_series = self._pad_series("dacs_series", 0.0)
        self.dacs_lo = self._pad_series("dacs_series_lo", np.nan)
        self.dacs_hi = self._pad_series("dacs_series_hi", np.nan)
        self.ce_series = self._pad_series("coverage_entropy_norm_series", np.nan)
        self.collapse_series = self._pad_series("collapse_rate_series", 0.0)
        
        # Load supplementary info
        self.pass_result, self.epa_actual, self.pass_length, self.pass_loc = self._load_pass_outcome()
        
    def _build_player_series(self) -> Tuple[List[PlayerSeries], Optional[PlayerSeries], Optional[PlayerSeries], int, str]:
        files_in = analytics_input_files(self.root_dir)
        parts = []
        for fp in files_in:
            for chunk in pd.read_csv(fp, chunksize=200_000):
                sel = chunk[(chunk["game_id"] == self.game_id) & (chunk["play_id"] == self.play_id)]
                if not sel.empty:
                    parts.append(sel)
        if not parts:
            raise FileNotFoundError(f"No input rows for game {self.game_id} play {self.play_id}")
        df_in = pd.concat(parts, ignore_index=True)
        frame0 = int(df_in["frame_id"].min())
        snap = df_in[df_in["frame_id"] == frame0].copy()
        play_direction = str(snap["play_direction"].iloc[0]) if "play_direction" in snap.columns else "right"

        output_files = analytics_output_files(self.root_dir)
        df_out = load_game_output_rows(output_files, self.game_id)
        df_out = df_out[df_out["play_id"] == self.play_id].copy()
        if df_out.empty:
            raise FileNotFoundError(f"No tracking output rows for game {self.game_id} play {self.play_id}")

        step_count = int(max(self.data.get("num_frames_output", 0), df_out["frame_id"].max()))
        step_count = max(step_count, 1)

        player_series = []
        wr_series = None
        qb_series = None

        for _, row in snap.iterrows():
            nfl_id = int(row["nfl_id"])
            x0, y0 = normalize_direction(play_direction, row["x"], row["y"])
            speed0 = float(row["s"]) if not pd.isna(row["s"]) else 0.0
            heading_deg = normalize_heading(play_direction, row["dir"]) if not pd.isna(row["dir"]) else 0.0
            
            trace = np.full((step_count, 2), np.nan, dtype=float)
            g = df_out[df_out["nfl_id"] == nfl_id]
            for _, out_row in g.iterrows():
                k = int(out_row["frame_id"]) - 1
                if 0 <= k < step_count:
                    trace[k] = normalize_direction(play_direction, out_row["x"], out_row["y"])

            role = str(row.get("player_role", ""))
            side = str(row.get("player_side", ""))
            
            # Assign colors based on role/side
            if role.lower() == "targeted receiver":
                color = COLOR_WR
            elif role.lower() == "passer":
                color = COLOR_QB
            elif side.lower() == "defense":
                color = COLOR_DEFENSE
            else:
                color = COLOR_OFFENSE

            ps = PlayerSeries(
                nfl_id=nfl_id,
                name=str(row.get("player_name", "")),
                position=str(row.get("player_position", "")),
                role=role,
                side=side,
                init_pos=np.array([x0, y0], dtype=float),
                speed=speed0,
                heading_deg=heading_deg,
                color=color,
                xy=trace,
            )

            if role.lower() == "targeted receiver":
                wr_series = ps
            elif role.lower() == "passer":
                qb_series = ps
            
            player_series.append(ps)

        return player_series, wr_series, qb_series, step_count, play_direction

    def _build_position_lookup(self) -> Dict[int, np.ndarray]:
        lookup = {}
        for ps in self.players:
            arr = ps.xy.copy()
            if arr.shape[0] < self.steps:
                arr = np.pad(arr, ((0, self.steps - arr.shape[0]), (0, 0)), constant_values=np.nan)
            lookup[ps.nfl_id] = arr
        return lookup

    def _pad_series(self, key: str, default: float) -> np.ndarray:
        arr = np.asarray(self.data.get(key, []), dtype=float)
        if arr.size == 0:
            arr = np.full(self.steps, default, dtype=float)
        if arr.size < self.steps:
            arr = np.pad(arr, (0, self.steps - arr.size), constant_values=default)
        else:
            arr = arr[:self.steps]
        return arr

    def _load_pass_outcome(self) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
        # Simplified loading logic
        supp_path = os.path.join(self.root_dir, "analytics", "data", "114239_nfl_competition_files_published_analytics_final", "supplementary_data.csv")
        if not os.path.exists(supp_path): return None, None, None, None
        
        # We can optimize this by caching if needed, but for single run it's fine
        # Or use the one from dacs_one_game if available? 
        # Let's just read specific rows to be safe and fast
        try:
            df = pd.read_csv(supp_path)
            row = df[(df["game_id"] == self.game_id) & (df["play_id"] == self.play_id)]
            if row.empty: return None, None, None, None
            r = row.iloc[0]
            return (
                str(r["pass_result"]) if not pd.isna(r["pass_result"]) else None,
                float(r["expected_points_added"]) if not pd.isna(r["expected_points_added"]) else None,
                str(r["pass_length"]) if not pd.isna(r["pass_length"]) else None,
                str(r["pass_location_type"]) if not pd.isna(r["pass_location_type"]) else None
            )
        except:
            return None, None, None, None

    def _setup_physics(self):
        params = self.data.get("params", {})
        self.a_max = float(params.get("a_max", 4.75))
        self.v_cap = float(params.get("v_cap", 8.2))
        self.a_lat = float(params.get("a_lat_max", DEFAULT_A_LAT_MAX))

    def _setup_corridor(self):
        self.corr_centers = list(self.data.get("corridor_centers", []))
        if not self.corr_centers:
            origin = self.players[0].init_pos if self.players else np.array([60.0, 26.65])
            self.corr_centers = [(float(origin[0]), float(origin[1]))] * self.steps
        else:
            last_center = self.corr_centers[-1]
            while len(self.corr_centers) < self.steps:
                self.corr_centers.append(last_center)
        
        self.corridor_radius = float(self.data.get("corridor_radius", CORRIDOR_RADIUS_YDS))
        rng = np.random.default_rng(self.seed)
        self.corridor_samples = [
            sample_points_in_disk(np.array(center, dtype=float), self.corridor_radius, self.samples, rng) 
            for center in self.corr_centers
        ]

    # -------------------------------------------------------------------------
    # Drawing Methods
    # -------------------------------------------------------------------------
    def draw_field(self, ax: plt.Axes):
        ax.set_facecolor(COLOR_FIELD)
        ax.set_xlim(FIELD_X_MIN, FIELD_X_MAX)
        ax.set_ylim(FIELD_Y_MIN, FIELD_Y_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        
        # Yard lines
        for x in np.arange(10, 111, 5):
            lw = 1.5 if x % 10 == 0 else 0.5
            alpha = 0.5 if x % 10 == 0 else 0.3
            ax.axvline(x, color=COLOR_GRID, lw=lw, alpha=alpha)
            
            # Numbers (simplified)
            if x % 10 == 0 and 10 < x < 110:
                num = x if x <= 50 else 120 - x
                ax.text(x - 1, 5, str(num), color=COLOR_GRID, fontsize=10, alpha=0.6, rotation=270, va='center')
                ax.text(x - 1, 53.3 - 5, str(num), color=COLOR_GRID, fontsize=10, alpha=0.6, rotation=90, va='center')

        # Endzones
        ax.add_patch(patches.Rectangle((0, FIELD_Y_MIN), 10, FIELD_Y_MAX - FIELD_Y_MIN, facecolor="#1b5e20", alpha=0.6))
        ax.add_patch(patches.Rectangle((110, FIELD_Y_MIN), 10, FIELD_Y_MAX - FIELD_Y_MIN, facecolor="#1b5e20", alpha=0.6))

    def draw_players(self, ax: plt.Axes, frame_idx: int, plotted_ids: Dict[int, np.ndarray]):
        for ps in self.players:
            # Skip if already plotted (e.g. WR/QB special handling)
            if ps.nfl_id in plotted_ids: continue
            
            pos = latest_position(self.positions[ps.nfl_id], frame_idx, ps.init_pos)
            
            # Trail
            # (Simplified: we can add trails if needed, but for clarity maybe just dots)
            
            # Dot
            ax.scatter(pos[0], pos[1], s=40, color=ps.color, edgecolor="white", linewidths=0.5, zorder=5)
            plotted_ids[ps.nfl_id] = pos
            
            # Jersey Number
            # ax.text(pos[0], pos[1], str(ps.nfl_id), fontsize=6, color='white', ha='center', va='center', zorder=6)

    def draw_special_roles(self, ax: plt.Axes, frame_idx: int, plotted_ids: Dict[int, np.ndarray]):
        if self.wr_series:
            wr_pos = latest_position(self.positions[self.wr_series.nfl_id], frame_idx, self.wr_series.init_pos)
            ax.scatter(wr_pos[0], wr_pos[1], s=70, color=COLOR_WR, edgecolor="white", linewidths=1.5, zorder=6)
            plotted_ids[self.wr_series.nfl_id] = wr_pos
            
        if self.qb_series:
            qb_pos = latest_position(self.positions[self.qb_series.nfl_id], min(frame_idx, self.positions[self.qb_series.nfl_id].shape[0]-1), self.qb_series.init_pos)
            ax.scatter(qb_pos[0], qb_pos[1], s=60, color=COLOR_QB, edgecolor="black", linewidths=1.0, zorder=5)
            plotted_ids[self.qb_series.nfl_id] = qb_pos

    def draw_defenders_and_reach(self, ax: plt.Axes, frame_idx: int, plotted_ids: Dict[int, np.ndarray], samples_now: np.ndarray, center_now: np.ndarray):
        t_now = (frame_idx + 1) * DT
        t_future = self.steps * DT
        
        coverage_mask = np.zeros(samples_now.shape[0], dtype=bool)
        future_mask = np.zeros(samples_now.shape[0], dtype=bool)
        
        for d in self.defenders:
            arr = self.positions.get(d.nfl_id, np.empty((0, 2)))
            pos = latest_position(arr, frame_idx, d.init_pos)
            heading = compute_heading(arr, frame_idx, math.radians(d.heading_deg))
            speed = compute_speed(arr, frame_idx, d.speed)
            
            # Draw Defender
            ax.scatter(pos[0], pos[1], s=50, color=COLOR_DEFENSE, edgecolor="white", linewidths=1.0, zorder=5)
            plotted_ids[d.nfl_id] = pos
            
            # Reach Logic
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            
            a_now = reachable_radius(t_now, speed, self.a_max, self.v_cap)
            b_lim_now = (max(speed, 1e-6)**2) / max(self.a_lat, 1e-6)
            b_now = min(a_now, b_lim_now)
            
            a_fut = reachable_radius(t_future, speed, self.a_max, self.v_cap)
            b_lim_fut = (max(speed, 1e-6)**2) / max(self.a_lat, 1e-6)
            b_fut = min(a_fut, b_lim_fut)
            
            dx = samples_now[:, 0] - pos[0]
            dy = samples_now[:, 1] - pos[1]
            xprime = dx * cos_h + dy * sin_h
            yprime = -dx * sin_h + dy * cos_h
            
            inside_now = (xprime / max(a_now, 1e-6))**2 + (yprime / max(b_now, 1e-6))**2 <= 1.0
            inside_fut = (xprime / max(a_fut, 1e-6))**2 + (yprime / max(b_fut, 1e-6))**2 <= 1.0
            
            coverage_mask |= inside_now
            future_mask |= inside_fut
            
        # Draw Samples
        uncovered = samples_now[~coverage_mask]
        covered = samples_now[coverage_mask]
        future = samples_now[future_mask]
        
        if uncovered.size > 0:
            ax.scatter(uncovered[:, 0], uncovered[:, 1], s=15, color="white", alpha=0.3, edgecolor="none", zorder=2)
        if covered.size > 0:
            ax.scatter(covered[:, 0], covered[:, 1], s=20, color=COLOR_DEFENSE, alpha=0.4, edgecolor="none", zorder=3)
        if future.size > 0:
            ax.scatter(future[:, 0], future[:, 1], s=15, facecolor="none", edgecolor=COLOR_FUTURE_REACH, alpha=0.5, zorder=3)

        # Draw Corridor
        ax.add_patch(patches.Circle((center_now[0], center_now[1]), self.corridor_radius, facecolor="white", alpha=0.1, edgecolor="white", ls="--", zorder=1))


    def create_animation(self, out_path: str):
        fig = plt.figure(figsize=(12, 8), dpi=self.dpi)
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[4, 1.2, 0.8], hspace=0.3)
        
        ax_field = fig.add_subplot(gs[0])
        ax_series = fig.add_subplot(gs[1])
        ax_metrics = fig.add_subplot(gs[2])
        
        # Static setup
        self.draw_field(ax_field)
        
        # Series Plot Setup
        t_axis = np.arange(1, self.steps + 1, dtype=float) * DT
        ax_series.set_facecolor("#f5f5f5")
        ax_series.set_xlim(0, t_axis[-1])
        ax_series.set_ylim(0, 100)
        ax_series.set_ylabel("DACS (%)", fontweight="bold", color=COLOR_DACS_LINE)
        ax_series.grid(True, color="#e0e0e0", linestyle="--")
        
        # Bands
        ax_series.fill_between(t_axis, self.dacs_lo, self.dacs_hi, color=COLOR_DACS_BAND, alpha=0.4, label="Uncertainty (5-95%)")
        line_dacs, = ax_series.plot(t_axis, self.dacs_series, color=COLOR_DACS_LINE, lw=2.5, label="DACS")
        
        # CE Twin Axis
        ax_ce = ax_series.twinx()
        line_ce, = ax_ce.plot(t_axis, self.ce_series, color=COLOR_CE_LINE, lw=1.5, ls="--", alpha=0.8, label="Entropy")
        ax_ce.set_ylim(0, 1)
        ax_ce.set_ylabel("Entropy", color=COLOR_CE_LINE)
        
        # Cursor
        cursor_line = ax_series.axvline(0, color="#333333", linestyle=":", alpha=0.8)
        cursor_point, = ax_series.plot([], [], "o", color=COLOR_DACS_LINE, ms=6)
        
        # Legend
        lines = [line_dacs, patches.Patch(facecolor=COLOR_DACS_BAND, alpha=0.4), line_ce]
        labels = ["DACS", "Uncertainty", "Entropy"]
        ax_series.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8)

        # Metrics Setup
        ax_metrics.axis("off")
        
        pass_outcome_displayed = False

        def animate(frame_idx):
            nonlocal pass_outcome_displayed
            
            # 1. Field Update
            ax_field.cla()
            self.draw_field(ax_field)
            
            center_now = np.array(self.corr_centers[min(frame_idx, len(self.corr_centers)-1)], dtype=float)
            samples_now = self.corridor_samples[min(frame_idx, len(self.corridor_samples)-1)]
            
            plotted_ids = {}
            self.draw_defenders_and_reach(ax_field, frame_idx, plotted_ids, samples_now, center_now)
            self.draw_special_roles(ax_field, frame_idx, plotted_ids)
            self.draw_players(ax_field, frame_idx, plotted_ids)
            
            # Outcome Text
            if not pass_outcome_displayed and frame_idx >= self.steps - 5 and self.pass_result:
                pass_outcome_displayed = True
            
            if pass_outcome_displayed:
                res_map = {"C": "COMPLETE", "I": "INCOMPLETE", "IN": "INTERCEPTION"}
                res_col = {"C": "#4caf50", "I": "#f44336", "IN": "#ff9800"}
                txt = res_map.get(self.pass_result, self.pass_result)
                col = res_col.get(self.pass_result, "white")
                ax_field.text(0.5, 0.5, txt, transform=ax_field.transAxes, 
                              ha="center", va="center", fontsize=24, fontweight="bold", color=col,
                              bbox=dict(facecolor="black", alpha=0.7, edgecolor=col, pad=10))

            # 2. Series Cursor Update
            t_now = (frame_idx + 1) * DT
            idx = min(frame_idx, len(self.dacs_series)-1)
            val = self.dacs_series[idx]
            
            cursor_line.set_xdata([t_now, t_now])
            cursor_point.set_data([t_now], [val])
            
            # 3. Metrics Text Update
            ax_metrics.clear()
            ax_metrics.axis("off")
            
            # Left Column: DACS Stats
            dacs_txt = f"DACS: {val:.1f}%"
            band_txt = f"Range: {self.dacs_lo[idx]:.1f}% - {self.dacs_hi[idx]:.1f}%"
            ce_txt = f"Entropy: {self.ce_series[idx]:.2f}"
            
            ax_metrics.text(0.05, 0.7, dacs_txt, transform=ax_metrics.transAxes, fontsize=14, fontweight="bold", color=COLOR_DACS_LINE)
            ax_metrics.text(0.05, 0.4, band_txt, transform=ax_metrics.transAxes, fontsize=10, color="#555555")
            ax_metrics.text(0.05, 0.2, ce_txt, transform=ax_metrics.transAxes, fontsize=10, color=COLOR_CE_LINE)
            
            # Center Column: Play Context
            qb_name = self.qb_series.name if self.qb_series else "Unknown QB"
            wr_name = self.wr_series.name if self.wr_series else "Unknown Target"
            ctx_txt = f"{qb_name} → {wr_name}"
            route_txt = f"Route: {self.pass_loc or '--'} {self.pass_length or '--'}"
            
            ax_metrics.text(0.5, 0.7, ctx_txt, transform=ax_metrics.transAxes, ha="center", fontsize=12, fontweight="bold")
            ax_metrics.text(0.5, 0.4, route_txt, transform=ax_metrics.transAxes, ha="center", fontsize=10, color="#555555")
            
            # Right Column: Probabilities
            probs = self.data.get("event_probabilities", {})
            p_catch = probs.get("catch", 0.0)
            p_inc = probs.get("incomplete", 0.0)
            p_int = probs.get("interception", 0.0)
            
            ax_metrics.text(0.95, 0.7, f"P(Catch): {p_catch:.2f}", transform=ax_metrics.transAxes, ha="right", fontsize=10, color="#4caf50")
            ax_metrics.text(0.95, 0.5, f"P(Inc): {p_inc:.2f}", transform=ax_metrics.transAxes, ha="right", fontsize=10, color="#f44336")
            ax_metrics.text(0.95, 0.3, f"P(Int): {p_int:.2f}", transform=ax_metrics.transAxes, ha="right", fontsize=10, color="#ff9800")

            return []

        anim = animation.FuncAnimation(fig, animate, frames=self.steps, interval=max(35, int(1000/self.fps)), blit=False)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        anim.save(out_path, writer=animation.PillowWriter(fps=self.fps))
        plt.close(fig)
        return out_path

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize Defensive Air Control")
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
    
    # Load JSON
    json_path = os.path.join(root_dir, args.json_dir, f"game_{args.game_id}_play_{args.play_id}.json")
    if not os.path.exists(json_path):
        # Fallback to test dir if needed, or just fail
        json_path = os.path.join(root_dir, "analytics", "outputs", "dacs_test", f"game_{args.game_id}_play_{args.play_id}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")
            
    with open(json_path, "r") as f:
        data = json.load(f)

    viz = DACSVisualizer(root_dir, args.game_id, args.play_id, data, args.fps, args.dpi, args.samples, args.seed)
    out_path = os.path.join(args.out_dir, f"game_{args.game_id}_play_{args.play_id}.gif")
    print(f"Generating animation to {out_path}...")
    viz.create_animation(out_path)
    print("Done.")

if __name__ == "__main__":
    main()
