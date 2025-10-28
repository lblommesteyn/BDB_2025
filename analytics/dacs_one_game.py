import os
import sys
import glob
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if __package__ is None or __package__ == '':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from analytics.residual_model import (
    ResidualReachModel,
    build_feature_vector,
    load_residual_model,
)

# -----------------------------
# Constants and calibration
# -----------------------------
FRAMERATE_HZ = 10.0  # Next Gen Stats typical rate
DT = 1.0 / FRAMERATE_HZ
MAX_STEPS_CAP = 120  # cap to avoid long loops

# Disk radius (yards) representing the ball cross-section corridor around the centerline
CORRIDOR_RADIUS_YDS = 1.0
# Number of Monte Carlo samples to estimate coverage per time step
SAMPLES_PER_T = 200
# Evaluate Player Share for top-K nearest defenders to landing point (speed-up)
PLAYER_SHARE_TOPK = 6

# Defaults if we cannot read eda_summary.json
DEFAULT_A_MAX = 4.75  # yards/s^2 (p95 accel from EDA)
DEFAULT_V_CAP = 8.20   # yards/s (p99 speed from EDA)
DEFAULT_A_LAT_MAX = 6.0  # yards/s^2 (approx lateral accel cap)

# -----------------------------
# Data schemas
# -----------------------------
USECOLS_IN = [
    'game_id','play_id','player_to_predict','nfl_id','frame_id','play_direction',
    'absolute_yardline_number','player_name','player_height','player_weight',
    'player_birth_date','player_position','player_side','player_role',
    'x','y','s','a','dir','o','num_frames_output','ball_land_x','ball_land_y'
]

DTYPES_IN = {
    'game_id': 'int64','play_id': 'int64','player_to_predict':'boolean','nfl_id':'int64','frame_id':'int16',
    'play_direction':'category','absolute_yardline_number':'float32','player_name':'object','player_height':'object',
    'player_weight':'float32','player_birth_date':'object','player_position':'category','player_side':'category','player_role':'category',
    'x':'float32','y':'float32','s':'float32','a':'float32','dir':'float32','o':'float32','num_frames_output':'int16','ball_land_x':'float32','ball_land_y':'float32'
}

SUPP_USECOLS = [
    'game_id','play_id','pass_result','route_of_targeted_receiver','team_coverage_type',
    'team_coverage_man_zone','dropback_type','dropback_distance','play_action','pass_length','pass_location_type',
    'expected_points_added'
]

_BASELINE_EPA_CACHE: Optional[pd.Series] = None
_BASELINE_EPA_MEAN: Optional[float] = None
_EPA_EVENT_MEANS: Optional[Dict[str, float]] = None

OUTPUT_USECOLS = ['game_id','play_id','nfl_id','frame_id','x','y']
OUTPUT_DTYPES = {
    'game_id':'int64','play_id':'int64','nfl_id':'int64','frame_id':'int16','x':'float32','y':'float32'
}

@dataclass
class Defender:
    nfl_id: int
    name: str
    pos: Tuple[float, float]
    speed: float  # yards/s at snapshot
    dir_deg: float  # heading in degrees at snapshot
    accel: float = 0.0
    position: str = ''
    role: str = ''

@dataclass
class PlaySnapshot:
    game_id: int
    play_id: int
    frame_id: int
    qb_id: Optional[int]
    qb_pos: Tuple[float, float]
    wr_id: Optional[int]
    wr_pos: Tuple[float, float]
    defenders: List[Defender]
    ball_land: Tuple[float, float]
    num_frames_output: int
    flipped: bool  # whether we flipped coords to make offense go +x
    play_direction: str

# -----------------------------
# Utility functions
# -----------------------------

def load_calibration(root_dir: str) -> Tuple[float, float]:
    """Load a_max and v_cap from eda_summary.json if available."""
    fp = os.path.join(root_dir, 'eda_summary.json')
    a_max, v_cap = DEFAULT_A_MAX, DEFAULT_V_CAP
    if os.path.exists(fp):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                eda = json.load(f)
            speed_q = eda.get('analytics_input', {}).get('speed_quantiles', {})
            accel_q = eda.get('analytics_input', {}).get('accel_quantiles', {})
            if 'p99' in speed_q:
                v_cap = float(speed_q['p99'])
            if 'p95' in accel_q:
                a_max = float(accel_q['p95'])
        except Exception:
            pass
    return a_max, v_cap


def analytics_input_files(root_dir: str) -> List[str]:
    base = os.path.join(root_dir, 'analytics', 'data', '114239_nfl_competition_files_published_analytics_final')
    files = sorted(glob.glob(os.path.join(base, 'train', 'input_2023_w*.csv')))
    if not files:
        raise FileNotFoundError(f"No analytics input CSVs found under {base}")
    return files


def analytics_output_files(root_dir: str) -> List[str]:
    base = os.path.join(root_dir, 'analytics', 'data', '114239_nfl_competition_files_published_analytics_final')
    files = sorted(glob.glob(os.path.join(base, 'train', 'output_2023_w*.csv')))
    if not files:
        raise FileNotFoundError(f"No analytics output CSVs found under {base}")
    return files


def _flip_xy_if_left(play_direction_val: str, x: float, y: float) -> Tuple[float, float]:
    """If play direction is left, flip x so that offense goes toward +x consistently."""
    try:
        s = str(play_direction_val).lower()
    except Exception:
        s = ''
    if s.startswith('left'):
        return 120.0 - float(x), float(y)
    return float(x), float(y)

def _flip_heading_if_left(play_direction_val: str, dir_deg: float) -> float:
    """Mirror heading over vertical axis if play direction is left (approx).
    If angle is degrees with 0 along +x and increasing CCW, mirroring x -> 120-x maps theta to (180 - theta).
    """
    try:
        s = str(play_direction_val).lower()
    except Exception:
        s = ''
    th = float(dir_deg)
    if s.startswith('left'):
        return (180.0 - th) % 360.0
    return th % 360.0


def load_game_rows(files: List[str], game_id: int) -> pd.DataFrame:
    """Load all rows for a specific game_id across all input CSVs via chunked reads."""
    dfs = []
    for f in files:
        for chunk in pd.read_csv(f, usecols=USECOLS_IN, dtype=DTYPES_IN, chunksize=200_000):
            g = chunk[chunk['game_id'] == game_id]
            if not g.empty:
                dfs.append(g)
    if not dfs:
        raise ValueError(f"game_id {game_id} not found in inputs")
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_supplementary_rows(root_dir: str, game_id: int) -> pd.DataFrame:
    supp_path = os.path.join(root_dir, 'analytics', 'data', '114239_nfl_competition_files_published_analytics_final', 'supplementary_data.csv')
    if not os.path.exists(supp_path):
        raise FileNotFoundError("supplementary_data.csv not found.")
    df = pd.read_csv(supp_path, usecols=SUPP_USECOLS, dtype={
        'game_id':'int64','play_id':'int64','pass_result':'category','route_of_targeted_receiver':'category',
        'team_coverage_type':'category','team_coverage_man_zone':'category','dropback_type':'category',
        'dropback_distance':'float32','play_action':'category','pass_length':'category','pass_location_type':'category',
        'expected_points_added':'float32'
    })
    return df[df['game_id'] == game_id].copy()


def baseline_epa_lookup(root_dir: str) -> pd.Series:
    """Return baseline EPA by (pass_length, pass_location_type) multi-index."""
    global _BASELINE_EPA_CACHE, _BASELINE_EPA_MEAN, _EPA_EVENT_MEANS
    if _BASELINE_EPA_CACHE is not None:
        return _BASELINE_EPA_CACHE
    supp_path = os.path.join(root_dir, 'analytics', 'data', '114239_nfl_competition_files_published_analytics_final', 'supplementary_data.csv')
    if not os.path.exists(supp_path):
        raise FileNotFoundError("supplementary_data.csv not found.")
    df = pd.read_csv(
        supp_path,
        usecols=['pass_length','pass_location_type','expected_points_added','pass_result'],
        dtype={'pass_length':'category','pass_location_type':'category','expected_points_added':'float32','pass_result':'category'}
    )
    grouped = df.groupby(['pass_length','pass_location_type'], observed=True)['expected_points_added'].mean()
    _BASELINE_EPA_CACHE = grouped
    _BASELINE_EPA_MEAN = float(df['expected_points_added'].mean())
    _EPA_EVENT_MEANS = df.groupby('pass_result', observed=True)['expected_points_added'].mean().to_dict()
    return grouped


def event_epa_means(root_dir: str) -> Dict[str, float]:
    """Return mean EPA for pass outcomes (Complete, Incomplete, Interception)."""
    if _EPA_EVENT_MEANS is None:
        baseline_epa_lookup(root_dir)
    return _EPA_EVENT_MEANS or {}


def load_game_output_rows(files: List[str], game_id: int) -> pd.DataFrame:
    """Load all post-throw trajectory rows for a specific game_id."""
    dfs = []
    for f in files:
        for chunk in pd.read_csv(f, usecols=OUTPUT_USECOLS, dtype=OUTPUT_DTYPES, chunksize=200_000):
            g = chunk[chunk['game_id'] == game_id]
            if not g.empty:
                dfs.append(g)
    if not dfs:
        # Return empty dataframe with expected columns if outputs missing
        return pd.DataFrame(columns=OUTPUT_USECOLS)
    return pd.concat(dfs, ignore_index=True)


def pick_snapshot_for_play(df_play: pd.DataFrame) -> Optional[PlaySnapshot]:
    """Pick the earliest frame in this play as the snapshot. Build roles and defenders."""
    # Choose earliest frame present for this play
    f0 = int(df_play['frame_id'].min())
    snap = df_play[df_play['frame_id'] == f0].copy()

    # landing point and horizon
    # use most common (should be constant within play)
    play_dir_val = snap['play_direction'].iloc[0] if 'play_direction' in snap.columns else ''
    raw_blx = float(snap['ball_land_x'].iloc[0])
    raw_bly = float(snap['ball_land_y'].iloc[0])
    blx, bly = _flip_xy_if_left(play_dir_val, raw_blx, raw_bly)

    # Use max horizon across rows at f0
    T_frames = int(snap['num_frames_output'].max())
    T_frames = int(np.clip(T_frames, 1, MAX_STEPS_CAP))

    qb = snap[snap['player_role'] == 'Passer']
    wr = snap[snap['player_role'] == 'Targeted Receiver']

    qb_id = int(qb['nfl_id'].iloc[0]) if not qb.empty else None
    if not qb.empty:
        qx, qy = _flip_xy_if_left(play_dir_val, float(qb['x'].iloc[0]), float(qb['y'].iloc[0]))
    else:
        qx, qy = _flip_xy_if_left(play_dir_val, float(snap['x'].mean()), float(snap['y'].mean()))
    qb_pos = (qx, qy)

    wr_id = int(wr['nfl_id'].iloc[0]) if not wr.empty else None
    if not wr.empty:
        wx, wy = _flip_xy_if_left(play_dir_val, float(wr['x'].iloc[0]), float(wr['y'].iloc[0]))
    else:
        wx, wy = _flip_xy_if_left(play_dir_val, float(snap['x'].mean()), float(snap['y'].mean()))
    wr_pos = (wx, wy)

    # defenders at f0
    defs = []
    d_rows = snap[snap['player_side'] == 'Defense']
    for _, r in d_rows.iterrows():
        dx, dy = _flip_xy_if_left(play_dir_val, float(r['x']), float(r['y']))
        ddir = float(r['dir']) if not pd.isna(r['dir']) else 0.0
        ddir = _flip_heading_if_left(play_dir_val, ddir)
        defs.append(Defender(
            nfl_id=int(r['nfl_id']),
            name=str(r.get('player_name', '')),
            pos=(dx, dy),
            speed=float(r['s']) if not pd.isna(r['s']) else 0.0,
            dir_deg=ddir,
            accel=float(r.get('a', 0.0)) if not pd.isna(r.get('a', np.nan)) else 0.0,
            position=str(r.get('player_position', '')) if 'player_position' in r else '',
            role=str(r.get('player_role', '')) if 'player_role' in r else '',
        ))

    if len(defs) == 0:
        return None

    return PlaySnapshot(
        game_id=int(snap['game_id'].iloc[0]),
        play_id=int(snap['play_id'].iloc[0]),
        frame_id=f0,
        qb_id=qb_id,
        qb_pos=qb_pos,
        wr_id=wr_id,
        wr_pos=wr_pos,
        defenders=defs,
        ball_land=(blx, bly),
        num_frames_output=T_frames,
        flipped=str(play_dir_val).lower().startswith('left'),
        play_direction=str(play_dir_val),
    )


# -----------------------------
# Geometry & physics
# -----------------------------

def solve_min_time_to_distance(d0: float, s0: float, a_max: float, v_cap: float) -> float:
    """Minimal time to cover distance d0 with initial speed s0, accel limit a_max, speed cap v_cap."""
    s0 = max(0.0, float(s0))
    a_max = max(1e-6, float(a_max))
    v_cap = max(1e-6, float(v_cap))

    # time to reach cap
    t_acc = max(0.0, (v_cap - s0) / a_max)
    d_acc = s0 * t_acc + 0.5 * a_max * t_acc * t_acc

    if d0 <= d_acc + 1e-9:
        # solve 0.5 a t^2 + s0 t - d0 = 0
        A = 0.5 * a_max
        B = s0
        C = -d0
        disc = B*B - 4*A*C
        t = (-B + math.sqrt(max(0.0, disc))) / (2*A)
        return max(0.0, t)
    else:
        d_rem = d0 - d_acc
        t_rem = d_rem / v_cap
        return t_acc + t_rem


def reachable_radius(t: float, s0: float, a_max: float, v_cap: float) -> float:
    """Distance reachable in time t with initial speed s0, acceleration a_max, capped by v_cap."""
    s0 = max(0.0, float(s0))
    t = max(0.0, float(t))
    a_max = max(1e-6, float(a_max))
    v_cap = max(1e-6, float(v_cap))

    # accelerate until cap or until time over
    t_acc = max(0.0, (v_cap - s0) / a_max)
    if t <= t_acc:
        return s0 * t + 0.5 * a_max * t * t
    else:
        d_acc = s0 * t_acc + 0.5 * a_max * t_acc * t_acc
        t_rem = t - t_acc
        return d_acc + v_cap * t_rem


def sample_points_in_disk(center: np.ndarray, radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    # Polar sampling with sqrt for uniform disk
    u = rng.random(n)
    r = radius * np.sqrt(u)
    theta = 2 * np.pi * rng.random(n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.stack([x, y], axis=1)


def dacs_time_series(snap: PlaySnapshot, a_max: float, v_cap: float,
                     dt: float = DT, samples_per_t: int = SAMPLES_PER_T,
                     corridor_radius: float = CORRIDOR_RADIUS_YDS,
                     topk_ps: int = PLAYER_SHARE_TOPK,
                     seed: int = 42,
                     residual_model: Optional[ResidualReachModel] = None,
                     play_output: Optional[pd.DataFrame] = None,
                     supplementary_row: Optional[Any] = None,
                     baseline_epa_map: Optional[Dict[Tuple[str, str], float]] = None,
                     baseline_epa_default: float = 0.0,
                     event_epa_map: Optional[Dict[str, float]] = None) -> Dict:
    rng = np.random.default_rng(seed)

    qb = np.array(snap.qb_pos, dtype=float)
    wr = np.array(snap.wr_pos, dtype=float)
    b = np.array(snap.ball_land, dtype=float)
    vec = b - qb
    L = float(np.linalg.norm(vec)) + 1e-9

    # times and centers along straight-line ball path
    steps = int(np.clip(snap.num_frames_output, 1, MAX_STEPS_CAP))
    times = np.arange(1, steps + 1, dtype=float) * dt
    total_time = times[-1] if times.size > 0 else 0.0

    # Precompute defender arrays
    D = len(snap.defenders)
    def_pos = np.array([d.pos for d in snap.defenders], dtype=float)  # (D,2)
    def_spd = np.array([max(0.0, d.speed) for d in snap.defenders], dtype=float)   # (D,)
    def_acc = np.array([float(getattr(d, 'accel', 0.0)) for d in snap.defenders], dtype=float)
    def_dir_deg = np.array([float(getattr(d, 'dir_deg', 0.0)) for d in snap.defenders], dtype=float)
    def_cos = np.cos(np.deg2rad(def_dir_deg))
    def_sin = np.sin(np.deg2rad(def_dir_deg))

    residual_std = None
    clip_min, clip_max = 0.0, 3.0
    if residual_model is not None:
        residual_std = np.asarray(residual_model.target_stats.get('resid_std', np.zeros(2)), dtype=float)
        clip_min, clip_max = residual_model.clip_bounds

    # Actual trajectory lookup for defenders and WR
    actual_traj_lookup: Dict[int, Dict[str, np.ndarray]] = {}
    if play_output is not None and not play_output.empty:
        play_output_local = play_output.copy()
        if snap.flipped:
            play_output_local['x'] = 120.0 - play_output_local['x']
        for pid, g in play_output_local.groupby('nfl_id'):
            g_valid = g[(g['frame_id'] >= 1) & (g['frame_id'] <= steps)]
            if g_valid.empty:
                continue
            frames_arr = g_valid['frame_id'].to_numpy(dtype=int)
            pos_arr = g_valid[['x','y']].to_numpy(dtype=float)
            actual_traj_lookup[int(pid)] = {'frames': frames_arr, 'pos': pos_arr}

    def build_series_for_player(nfl_id: Optional[int], init_pos: np.ndarray) -> np.ndarray:
        series = np.full((steps + 1, 2), np.nan, dtype=float)
        series[0] = init_pos
        if nfl_id is not None:
            traj = actual_traj_lookup.get(int(nfl_id))
            if traj is not None:
                for frame, pos in zip(traj['frames'], traj['pos']):
                    if 1 <= frame <= steps:
                        series[frame] = pos
        for i in range(1, steps + 1):
            if np.any(np.isnan(series[i])):
                series[i] = series[i-1]
        return series

    wr_series = build_series_for_player(snap.wr_id, wr)
    defender_series = {
        int(d.nfl_id): build_series_for_player(d.nfl_id, def_pos[idx])
        for idx, d in enumerate(snap.defenders)
    }

    pursuit_eff: Dict[int, float] = {}
    hawk_reaction: Dict[int, float] = {}
    actual_arrival: Dict[int, float] = {}
    path_lengths: Dict[int, float] = {}

    arrival_threshold = max(corridor_radius, 1.0)
    for idx, d in enumerate(snap.defenders):
        series = defender_series.get(int(d.nfl_id))
        if series is None:
            pursuit_eff[int(d.nfl_id)] = math.nan
            hawk_reaction[int(d.nfl_id)] = math.nan
            actual_arrival[int(d.nfl_id)] = math.nan
            continue
        has_traj = int(d.nfl_id) in actual_traj_lookup
        if not has_traj:
            pursuit_eff[int(d.nfl_id)] = math.nan
            hawk_reaction[int(d.nfl_id)] = math.nan
            actual_arrival[int(d.nfl_id)] = math.nan
            continue

        diffs_seg = np.diff(series, axis=0)
        seg_len = np.linalg.norm(diffs_seg, axis=1)
        L_actual = float(np.sum(seg_len))
        path_lengths[int(d.nfl_id)] = L_actual
        L_opt = float(np.linalg.norm(b - def_pos[idx]))
        if L_actual <= 1e-6:
            pursuit_eff[int(d.nfl_id)] = 1.0 if L_opt <= 1e-6 else 0.0
        else:
            pursuit_eff[int(d.nfl_id)] = float(np.clip(L_opt / max(L_actual, 1e-6), 0.0, 1.0))

        dist_ball = np.linalg.norm(series - b[None, :], axis=1)
        arrival_idx = np.where(dist_ball <= arrival_threshold)[0]
        actual_arrival[int(d.nfl_id)] = float(arrival_idx[0] * dt) if arrival_idx.size > 0 else math.nan

        dist_wr = np.linalg.norm(series - wr_series, axis=1)
        diffs_wr = np.diff(dist_wr)
        win = 2
        hrt_val = math.nan
        for frame in range(win, len(dist_wr)):
            if np.all(diffs_wr[frame - win:frame] < -1e-4):
                hrt_val = frame * dt
                break
        hawk_reaction[int(d.nfl_id)] = hrt_val

    # Select top-K defenders by proximity to landing to compute Player Share
    dists_to_b = np.linalg.norm(def_pos - b[None, :], axis=1)
    idx_ps = np.argsort(dists_to_b)[:min(topk_ps, D)]

    dacs_vals = []
    dacs_frac_vals = []
    collapse_vals = []
    dacs_low_vals = []
    dacs_high_vals = []
    ps_at_T = {}
    ce_vals = []
    ce_norm_vals = []

    # For CTS and LII, compute once
    CTS = {}
    CTS_spec = {}
    LII = {}

    # LII per defender using WR position if available
    wr = np.array(snap.wr_pos, dtype=float)
    wr_to_b = b - wr
    for j, d in enumerate(snap.defenders):
        db_to_wr = wr - np.array(d.pos, dtype=float)
        # avoid zero-length vectors
        if np.linalg.norm(db_to_wr) < 1e-6 or np.linalg.norm(wr_to_b) < 1e-6:
            LII[d.nfl_id] = np.nan
        else:
            cosang = float(np.dot(db_to_wr, wr_to_b) / (np.linalg.norm(db_to_wr) * np.linalg.norm(wr_to_b)))
            LII[d.nfl_id] = np.clip(cosang, -1.0, 1.0)

    # CTS based on minimal arrival time vs T
    T_total = times[-1]
    for j, d in enumerate(snap.defenders):
        d0 = float(np.linalg.norm(b - np.array(d.pos)))
        t_i = solve_min_time_to_distance(d0, d.speed, a_max, v_cap)
        # If can't reach by T, remaining distance at T
        rT = reachable_radius(T_total, d.speed, a_max, v_cap)
        d_rem = max(0.0, d0 - rT)
        # params for heuristic CTS (kept as alternative)
        r_scale = 2.0   # yards scale for distance penalty
        tau = 0.5       # seconds scale for timing penalty
        CTS[d.nfl_id] = math.exp(-d_rem / r_scale) * math.exp(-abs(t_i - T_total) / tau)
        # Spec CTS: 1 - |t_def - T| / T, clipped to [0,1]
        CTS_spec[d.nfl_id] = max(0.0, 1.0 - abs(t_i - T_total) / max(T_total, 1e-9))

    # Precompute distance of each defender to the throw line segment for PS windowing (~12 yds)
    PS_WINDOW_YDS = 12.0
    # vector along path and its squared length
    ab = vec
    L2 = float(np.dot(ab, ab))
    d_line = np.zeros((D,), dtype=float)
    for j in range(D):
        p = def_pos[j]
        ap = p - qb
        tseg = 0.0 if L2 <= 1e-9 else float(np.clip(np.dot(ap, ab) / L2, 0.0, 1.0))
        nearest = qb + tseg * ab
        d_line[j] = float(np.linalg.norm(p - nearest))
    within_window = d_line <= PS_WINDOW_YDS

    # Main loop for DACS
    corridor_centers: List[Tuple[float,float]] = []
    for k, t in enumerate(times, start=1):
        alpha = float(k) / float(steps)
        center = qb + alpha * vec  # straight-line interpolation
        corridor_centers.append((float(center[0]), float(center[1])))

        # Sample corridor points
        P = sample_points_in_disk(center, corridor_radius, samples_per_t, rng)  # (M,2)

        # Oriented ellipse reach per defender at horizon t
        base_a_axis = def_spd * t + 0.5 * a_max * (t * t)   # (D,)
        b_limit = np.where(DEFAULT_A_LAT_MAX > 1e-9, (np.maximum(def_spd, 1e-6) ** 2) / DEFAULT_A_LAT_MAX, base_a_axis)
        base_b_axis = np.minimum(base_a_axis, b_limit)           # (D,)

        if residual_model is not None and D > 0:
            feat_mat = np.vstack([
                build_feature_vector(
                    t=float(t),
                    total_time=total_time,
                    defender=d,
                    qb_pos=qb,
                    wr_pos=wr,
                    ball_land=b,
                )
                for d in snap.defenders
            ])
            scales = residual_model.predict_scales(feat_mat)
            long_scale = scales[:, 0]
            lat_scale = scales[:, 1]
            if residual_std is not None:
                long_scale_low = np.clip(long_scale - residual_std[0], clip_min, clip_max)
                long_scale_high = np.clip(long_scale + residual_std[0], clip_min, clip_max)
                lat_scale_low = np.clip(lat_scale - residual_std[1], clip_min, clip_max)
                lat_scale_high = np.clip(lat_scale + residual_std[1], clip_min, clip_max)
            else:
                long_scale_low = long_scale_high = long_scale
                lat_scale_low = lat_scale_high = lat_scale
        else:
            long_scale = np.ones_like(base_a_axis)
            lat_scale = np.ones_like(base_b_axis)
            long_scale_low = long_scale_high = long_scale
            lat_scale_low = lat_scale_high = lat_scale

        a_axis = np.maximum(base_a_axis * long_scale, 1e-6)
        b_axis = np.maximum(base_b_axis * lat_scale, 1e-6)
        a_axis_low = np.maximum(base_a_axis * long_scale_low, 1e-6)
        b_axis_low = np.maximum(base_b_axis * lat_scale_low, 1e-6)
        a_axis_high = np.maximum(base_a_axis * long_scale_high, 1e-6)
        b_axis_high = np.maximum(base_b_axis * lat_scale_high, 1e-6)

        # Vector from defender to samples
        diff = P[None, :, :] - def_pos[:, None, :]     # (D, M, 2)
        dx = diff[:, :, 0]
        dy = diff[:, :, 1]
        # Rotate into defender's frame (x' along heading)
        xprime = dx * def_cos[:, None] + dy * def_sin[:, None]     # (D, M)
        yprime = -dx * def_sin[:, None] + dy * def_cos[:, None]    # (D, M)
        # Ellipse test: (x'/a)^2 + (y'/b)^2 <= 1 (guard zero)
        aa = np.maximum(a_axis, 1e-6)[:, None]
        bb = np.maximum(b_axis, 1e-6)[:, None]
        inside = (xprime / aa) ** 2 + (yprime / bb) ** 2 <= 1.0    # (D, M)
        if residual_model is not None:
            aa_low = np.maximum(a_axis_low, 1e-6)[:, None]
            bb_low = np.maximum(b_axis_low, 1e-6)[:, None]
            aa_high = np.maximum(a_axis_high, 1e-6)[:, None]
            bb_high = np.maximum(b_axis_high, 1e-6)[:, None]
            inside_low = (xprime / aa_low) ** 2 + (yprime / bb_low) ** 2 <= 1.0
            inside_high = (xprime / aa_high) ** 2 + (yprime / bb_high) ** 2 <= 1.0
        else:
            inside_low = inside_high = inside

        # Coverage boolean: any defender ellipse contains the sample
        cover_mat = inside
        covered = cover_mat.any(axis=0)             # (M,)
        dacs_frac = float(covered.mean())
        dacs = dacs_frac * 100.0
        dacs_frac_vals.append(dacs_frac)
        dacs_vals.append(dacs)
        if residual_model is not None:
            covered_low = inside_low.any(axis=0)
            covered_high = inside_high.any(axis=0)
            dacs_low_vals.append(float(covered_low.mean()) * 100.0)
            dacs_high_vals.append(float(covered_high.mean()) * 100.0)
        else:
            dacs_low_vals.append(dacs)
            dacs_high_vals.append(dacs)

        # Collapse rate (finite diff)
        if len(dacs_vals) == 1:
            collapse_vals.append(0.0)
        else:
            collapse_vals.append((dacs_vals[-1] - dacs_vals[-2]) / dt)

        # Coverage Entropy: assign each covered sample to its nearest covering defender
        if covered.any():
            # Euclidean distances from defenders to samples
            dist_euclid = np.sqrt(dx*dx + dy*dy)  # (D, M)
            # Mask distances where defender cannot reach (outside ellipse)
            dist_masked = np.where(cover_mat, dist_euclid, np.inf)  # (D,M)
            # For covered samples, find nearest covering defender index
            try:
                nearest = np.argmin(dist_masked[:, covered], axis=0)  # (#covered,)
            except ValueError:
                nearest = np.array([], dtype=int)
            if nearest.size > 0:
                unique, counts = np.unique(nearest, return_counts=True)
                p = counts.astype(float) / counts.sum()
                ce = float(-(p * np.log(p + 1e-12)).sum())
                # Normalize by log(D) to be in ~[0,1]
                ce_norm = float(ce / max(1e-9, math.log(max(2, D)))) if D > 1 else 0.0
                ce_vals.append(ce)
                ce_norm_vals.append(ce_norm)
            else:
                ce_vals.append(0.0)
                ce_norm_vals.append(0.0)
        else:
            ce_vals.append(0.0)
            ce_norm_vals.append(0.0)

        # Player Share at final step only (reduce compute)
        if k == steps:
            # Precompute coverage by all others for each i in idx_ps
            baseline_cov = covered.copy()
            for i in idx_ps:
                # remove defender i's ellipse and recompute union boolean
                covered_without_i = np.delete(cover_mat, i, axis=0).any(axis=0)
                ps = float(baseline_cov.mean() - covered_without_i.mean()) * 100.0
                ps_at_T[int(snap.defenders[i].nfl_id)] = max(0.0, ps)

    # Normalize Player Share over defenders near the flight path
    player_share_norm_at_T = {}
    if ps_at_T:
        # select defenders within window and present in ps_at_T
        selected_ids = [int(snap.defenders[j].nfl_id) for j in range(D) if within_window[j]]
        keys_in_window = [pid for pid in ps_at_T.keys() if pid in selected_ids]
        if not keys_in_window:
            keys_in_window = list(ps_at_T.keys())
        pos_vals = {pid: max(0.0, float(ps_at_T.get(pid, 0.0))) for pid in keys_in_window}
        total_pos = float(sum(pos_vals.values()))
        if total_pos > 1e-9:
            for pid in keys_in_window:
                player_share_norm_at_T[pid] = float(pos_vals[pid] / total_pos)
        else:
            for pid in keys_in_window:
                player_share_norm_at_T[pid] = 0.0

    dacs_frac_arr = np.array(dacs_frac_vals, dtype=float)
    dacs_low_arr = np.array(dacs_low_vals, dtype=float)
    dacs_high_arr = np.array(dacs_high_vals, dtype=float)
    ce_norm_arr = np.array(ce_norm_vals, dtype=float)

    bfoi = float(np.trapz(dacs_frac_arr, dx=dt) / max(total_time, 1e-6)) if dacs_frac_arr.size > 0 else 0.0
    dacs_final_frac = float(dacs_frac_arr[-1]) if dacs_frac_arr.size > 0 else 0.0
    dacs_final = float(dacs_vals[-1]) if dacs_vals else 0.0
    dacs_final_lo = float(dacs_low_arr[-1]) if dacs_low_arr.size > 0 else dacs_final
    dacs_final_hi = float(dacs_high_arr[-1]) if dacs_high_arr.size > 0 else dacs_final
    ce_final_norm = float(ce_norm_arr[-1]) if ce_norm_arr.size > 0 else 0.0

    cts_spec_max = max(CTS_spec.values()) if CTS_spec else 0.0
    coverage_intensity = float(np.clip(
        0.65 * dacs_final_frac + 0.25 * (1.0 - cts_spec_max) + 0.10 * (1.0 - ce_final_norm),
        0.0,
        1.2
    ))

    probs = np.array([
        max(0.0, 1.0 - 0.75 * coverage_intensity),   # catch
        max(0.0, 0.80 * coverage_intensity),         # incompletion/drop
        max(0.0, 0.20 * coverage_intensity)          # interception
    ], dtype=float)
    total_prob = probs.sum()
    if total_prob > 1e-6:
        probs /= total_prob
    p_catch, p_drop, p_int = probs.tolist()
    dvi = float(np.var(probs))

    event_epa_map = event_epa_map or {}
    epa_catch = float(event_epa_map.get('C', baseline_epa_default))
    epa_drop = float(event_epa_map.get('I', 0.0))
    epa_int = float(event_epa_map.get('IN', -2.5))
    expected_epa_cov = float(p_catch * epa_catch + p_drop * epa_drop + p_int * epa_int)

    pass_length_val = ''
    pass_loc_val = ''
    pass_result_val = ''
    baseline_epa = baseline_epa_default
    actual_epa = baseline_epa_default
    if supplementary_row is not None:
        pass_length_val = str(getattr(supplementary_row, 'pass_length', '') or '')
        pass_loc_val = str(getattr(supplementary_row, 'pass_location_type', '') or '')
        key = (pass_length_val, pass_loc_val)
        if baseline_epa_map is not None:
            baseline_epa = float(baseline_epa_map.get(key, baseline_epa_default))
        actual_epa_val = getattr(supplementary_row, 'expected_points_added', baseline_epa)
        try:
            actual_epa = float(actual_epa_val)
        except (TypeError, ValueError):
            actual_epa = baseline_epa
        if math.isnan(actual_epa):
            actual_epa = baseline_epa
        pass_result_val = str(getattr(supplementary_row, 'pass_result', ''))

    eaepa_model = float(baseline_epa - expected_epa_cov)
    eaepa_realized = float(baseline_epa - actual_epa)

    pursuit_vals = [v for v in pursuit_eff.values() if not math.isnan(v)]
    pursuit_mean = float(np.mean(pursuit_vals)) if pursuit_vals else math.nan
    pursuit_min = float(np.min(pursuit_vals)) if pursuit_vals else math.nan
    hrt_vals = [v for v in hawk_reaction.values() if not math.isnan(v)]
    hrt_mean = float(np.mean(hrt_vals)) if hrt_vals else math.nan
    hrt_min = float(np.min(hrt_vals)) if hrt_vals else math.nan
    arrival_vals = [v for v in actual_arrival.values() if not math.isnan(v)]
    arrival_mean = float(np.mean(arrival_vals)) if arrival_vals else math.nan
    arrival_min = float(np.min(arrival_vals)) if arrival_vals else math.nan

    event_probabilities = {
        'catch': float(p_catch),
        'incomplete': float(p_drop),
        'interception': float(p_int),
    }

    pursuit_summary_dict = {
        'mean': None if math.isnan(pursuit_mean) else pursuit_mean,
        'min': None if math.isnan(pursuit_min) else pursuit_min,
    }
    hrt_summary_dict = {
        'mean': None if math.isnan(hrt_mean) else hrt_mean,
        'min': None if math.isnan(hrt_min) else hrt_min,
    }
    arrival_summary_dict = {
        'mean': None if math.isnan(arrival_mean) else arrival_mean,
        'min': None if math.isnan(arrival_min) else arrival_min,
    }

    out = {
        'game_id': snap.game_id,
        'play_id': snap.play_id,
        'frame_id': snap.frame_id,
        'qb_id': snap.qb_id,
        'wr_id': snap.wr_id,
        'n_defenders': len(snap.defenders),
        'corridor_length': float(L),
        'num_frames_output': int(snap.num_frames_output),
        'dt': float(dt),
        'dacs_frac_series': dacs_frac_vals,          # 0..1
        'dacs_series': dacs_vals,                   # %
        'dacs_series_lo': dacs_low_vals,
        'dacs_series_hi': dacs_high_vals,
        'collapse_rate_series': collapse_vals,      # % per second
        'player_share_at_T': ps_at_T,               # % contribution at catch
        'player_share_norm_at_T': player_share_norm_at_T,  # normalized shares over window
        'CTS': CTS,
        'CTS_spec': CTS_spec,
        'LII': LII,
        'coverage_entropy_series': ce_vals,
        'coverage_entropy_norm_series': ce_norm_vals,
        'corridor_centers': corridor_centers,
        'corridor_radius': float(corridor_radius),
        'flipped': bool(snap.flipped),
        'pursuit_eff': pursuit_eff,
        'pursuit_summary': pursuit_summary_dict,
        'hawk_reaction_time': hawk_reaction,
        'hawk_reaction_summary': hrt_summary_dict,
        'actual_arrival_time': actual_arrival,
        'arrival_summary': arrival_summary_dict,
        'path_lengths': path_lengths,
        'bfoi': bfoi,
        'dacs_final': dacs_final,
        'dacs_final_lo': dacs_final_lo,
        'dacs_final_hi': dacs_final_hi,
        'coverage_intensity': coverage_intensity,
        'event_probabilities': event_probabilities,
        'baseline_epa': baseline_epa,
        'expected_epa_coverage': expected_epa_cov,
        'actual_epa': actual_epa,
        'eaepa_model': eaepa_model,
        'eaepa_realized': eaepa_realized,
        'dvi': dvi,
        'pass_length': pass_length_val,
        'pass_location_type': pass_loc_val,
        'pass_result': pass_result_val,
        'params': {
            'a_max': float(a_max),
            'v_cap': float(v_cap),
            'samples_per_t': int(samples_per_t),
            'corridor_radius': float(corridor_radius),
        }
    }
    return out


# -----------------------------
# Main pipeline
# -----------------------------

def compute_dacs_for_game(root_dir: str, game_id: int, out_dir: str,
                           samples_per_t: int = SAMPLES_PER_T,
                           corridor_radius: float = CORRIDOR_RADIUS_YDS,
                           topk_ps: int = PLAYER_SHARE_TOPK,
                           seed: int = 42,
                           residual_model_path: Optional[str] = None,
                           use_residual_model: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    a_max, v_cap = load_calibration(root_dir)
    files = analytics_input_files(root_dir)
    output_files = analytics_output_files(root_dir)
    df_game = load_game_rows(files, game_id)
    df_outputs = load_game_output_rows(output_files, game_id)
    supp_df = load_supplementary_rows(root_dir, game_id)
    baseline_epa_series = baseline_epa_lookup(root_dir)
    baseline_epa_map = baseline_epa_series.to_dict()
    baseline_epa_default = _BASELINE_EPA_MEAN if _BASELINE_EPA_MEAN is not None else 0.0
    event_epa_map = event_epa_means(root_dir)
    outputs_by_play: Dict[int, pd.DataFrame] = {}
    if not df_outputs.empty:
        for pid, g in df_outputs.groupby('play_id'):
            outputs_by_play[int(pid)] = g.copy()

    # Build snapshots per play
    groups = df_game.groupby('play_id')
    snapshots: List[PlaySnapshot] = []
    for play_id, g in groups:
        snap = pick_snapshot_for_play(g)
        if snap is not None:
            snapshots.append(snap)

    if not snapshots:
        raise ValueError(f"No valid play snapshots for game_id {game_id}")

    os.makedirs(out_dir, exist_ok=True)
    per_play_json_paths: List[str] = []
    ts_rows: List[Dict] = []

    # Results summary rows
    rows = []
    supp_lookup = {
        int(r.play_id): r for r in supp_df.itertuples()
    } if not supp_df.empty else {}

    residual_model: Optional[ResidualReachModel] = None
    if use_residual_model:
        model_path = residual_model_path or os.path.join(root_dir, 'analytics', 'models', 'residual_model.joblib')
        residual_model = load_residual_model(model_path)
        if residual_model is None:
            print(f"[WARN] Residual model not found at {model_path}; using physics-only reach.")

    for snap in snapshots:
        play_output = outputs_by_play.get(snap.play_id)
        supp_row = supp_lookup.get(snap.play_id)
        result = dacs_time_series(
            snap, a_max=a_max, v_cap=v_cap, dt=DT,
            samples_per_t=samples_per_t, corridor_radius=corridor_radius,
            topk_ps=topk_ps, seed=seed, residual_model=residual_model, play_output=play_output,
            supplementary_row=supp_row,
            baseline_epa_map=baseline_epa_map,
            baseline_epa_default=baseline_epa_default,
            event_epa_map=event_epa_map,
        )

        # Save per-play JSON
        jpath = os.path.join(out_dir, f"game_{snap.game_id}_play_{snap.play_id}.json")
        with open(jpath, 'w', encoding='utf-8') as f:
            json.dump(result, f)
        per_play_json_paths.append(jpath)

        # Derive summary stats
        dacs_series = np.array(result['dacs_series'], dtype=float)
        dacs_series_lo = np.array(result.get('dacs_series_lo', result['dacs_series']), dtype=float)
        dacs_series_hi = np.array(result.get('dacs_series_hi', result['dacs_series']), dtype=float)
        collapse_series = np.array(result['collapse_rate_series'], dtype=float)
        dacs_mean = float(dacs_series.mean())
        dacs_max = float(dacs_series.max())
        dacs_final = float(dacs_series[-1])
        dacs_final_lo = float(dacs_series_lo[-1]) if dacs_series_lo.size > 0 else dacs_final
        dacs_final_hi = float(dacs_series_hi[-1]) if dacs_series_hi.size > 0 else dacs_final
        peak_collapse = float(collapse_series.max()) if collapse_series.size > 0 else 0.0
        time_to_50 = np.nan
        idx_50 = np.where(dacs_series >= 50.0)[0]
        if idx_50.size > 0:
            time_to_50 = float((idx_50[0] + 1) * DT)

        # CE summary (normalized)
        ce_series = np.array(result.get('coverage_entropy_norm_series', []), dtype=float)
        ce_mean = float(np.nanmean(ce_series)) if ce_series.size > 0 else np.nan
        ce_final = float(ce_series[-1]) if ce_series.size > 0 else np.nan

        # top contributor at T
        ps_items = list(result['player_share_at_T'].items())
        top_ps_id, top_ps_val = (None, 0.0)
        if ps_items:
            top_ps_id, top_ps_val = max(ps_items, key=lambda kv: kv[1])

        bfoi = float(result.get('bfoi', np.nan))
        coverage_intensity = float(result.get('coverage_intensity', np.nan))
        eaepa_model = float(result.get('eaepa_model', np.nan))
        eaepa_realized = float(result.get('eaepa_realized', np.nan))
        dvi = float(result.get('dvi', np.nan))
        baseline_epa_play = float(result.get('baseline_epa', np.nan))
        expected_epa_cov = float(result.get('expected_epa_coverage', np.nan))
        actual_epa_play = float(result.get('actual_epa', np.nan))
        event_probs = result.get('event_probabilities', {})
        prob_catch = float(event_probs.get('catch', np.nan))
        prob_incomplete = float(event_probs.get('incomplete', np.nan))
        prob_int = float(event_probs.get('interception', np.nan))
        pursuit_summary = result.get('pursuit_summary', {})
        pursuit_mean_val = pursuit_summary.get('mean')
        pursuit_min_val = pursuit_summary.get('min')
        pursuit_mean = float(pursuit_mean_val) if pursuit_mean_val is not None else np.nan
        pursuit_min = float(pursuit_min_val) if pursuit_min_val is not None else np.nan
        hrt_summary = result.get('hawk_reaction_summary', {})
        hrt_mean_val = hrt_summary.get('mean')
        hrt_min_val = hrt_summary.get('min')
        hrt_mean = float(hrt_mean_val) if hrt_mean_val is not None else np.nan
        hrt_min = float(hrt_min_val) if hrt_min_val is not None else np.nan
        arrival_summary = result.get('arrival_summary', {})
        arrival_mean_val = arrival_summary.get('mean')
        arrival_min_val = arrival_summary.get('min')
        arrival_mean = float(arrival_mean_val) if arrival_mean_val is not None else np.nan
        arrival_min = float(arrival_min_val) if arrival_min_val is not None else np.nan
        pass_result_val = result.get('pass_result', '')
        pass_length_val = result.get('pass_length', '')
        pass_loc_val = result.get('pass_location_type', '')

        rows.append({
            'game_id': snap.game_id,
            'play_id': snap.play_id,
            'n_defenders': len(snap.defenders),
            'num_frames_output': snap.num_frames_output,
            'corridor_len': result['corridor_length'],
            'dacs_mean': dacs_mean,
            'dacs_max': dacs_max,
            'dacs_final': dacs_final,
            'dacs_final_lo': dacs_final_lo,
            'dacs_final_hi': dacs_final_hi,
            'peak_collapse_rate': peak_collapse,
            'time_to_50pct': time_to_50,
            'top_contributor_nfl_id': top_ps_id,
            'top_contributor_ps_pct': top_ps_val,
            'ce_mean_norm': ce_mean,
            'ce_final_norm': ce_final,
            'bfoi': bfoi,
            'coverage_intensity': coverage_intensity,
            'eaepa_model': eaepa_model,
            'eaepa_realized': eaepa_realized,
            'baseline_epa': baseline_epa_play,
            'expected_epa_coverage': expected_epa_cov,
            'actual_epa': actual_epa_play,
            'dvi': dvi,
            'prob_catch': prob_catch,
            'prob_incomplete': prob_incomplete,
            'prob_interception': prob_int,
            'pursuit_eff_mean': pursuit_mean,
            'pursuit_eff_min': pursuit_min,
            'hrt_mean': hrt_mean,
            'hrt_min': hrt_min,
            'arrival_mean': arrival_mean,
            'arrival_min': arrival_min,
            'pass_result': pass_result_val,
            'pass_length': pass_length_val,
            'pass_location_type': pass_loc_val,
        })

        # accumulate timeseries rows
        dacs_frac_series = result.get('dacs_frac_series', [])
        dacs_series_lo_list = result.get('dacs_series_lo', result['dacs_series'])
        dacs_series_hi_list = result.get('dacs_series_hi', result['dacs_series'])
        coverage_entropy_norm_series = result.get('coverage_entropy_norm_series', [])
        for k, t in enumerate(result['dacs_series'], start=1):
            ts_rows.append({
                'game_id': snap.game_id,
                'play_id': snap.play_id,
                'k': k,
                't': k * DT,
                'dacs': float(result['dacs_series'][k-1]),
                'dacs_frac': float(dacs_frac_series[k-1]) if k-1 < len(dacs_frac_series) else (float(result['dacs_series'][k-1]) / 100.0),
                'dacs_lo': float(dacs_series_lo_list[k-1]) if k-1 < len(dacs_series_lo_list) else float(result['dacs_series'][k-1]),
                'dacs_hi': float(dacs_series_hi_list[k-1]) if k-1 < len(dacs_series_hi_list) else float(result['dacs_series'][k-1]),
                'collapse_rate': float(result['collapse_rate_series'][k-1]),
                'ce_norm': float(coverage_entropy_norm_series[k-1]) if k-1 < len(coverage_entropy_norm_series) else np.nan,
            })

    df_summary = pd.DataFrame(rows)

    # Save summary CSV
    csv_path = os.path.join(out_dir, f"game_{game_id}_dacs_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # Persist physics bounds used
    bounds_path = os.path.join(out_dir, f"game_{game_id}_physics_bounds.json")
    with open(bounds_path, 'w', encoding='utf-8') as f:
        json.dump({'a_max': a_max, 'v_cap': v_cap}, f)

    # Persist timeseries
    if ts_rows:
        df_ts = pd.DataFrame(ts_rows)
        ts_path_parquet = os.path.join(out_dir, f"game_{game_id}_dacs_timeseries.parquet")
        ts_path_csv = os.path.join(out_dir, f"game_{game_id}_dacs_timeseries.csv")
        wrote_parquet = False
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401
            df_ts.to_parquet(ts_path_parquet, index=False)
            wrote_parquet = True
        except Exception:
            pass
        if not wrote_parquet:
            df_ts.to_csv(ts_path_csv, index=False)

    # Basic QA report
    qa = {}
    try:
        # monotonicity: fraction of decreases across all plays
        decreases = []
        for _, g in df_summary.iterrows():
            pjson = os.path.join(out_dir, f"game_{int(g['game_id'])}_play_{int(g['play_id'])}.json")
            with open(pjson, 'r', encoding='utf-8') as f:
                j = json.load(f)
            s = np.array(j['dacs_series'], dtype=float)
            if s.size > 1:
                diffs = np.diff(s)
                decreases.append(float(np.mean(diffs < -1e-6)))
        qa['frac_frames_with_dacs_decrease_mean'] = float(np.mean(decreases)) if decreases else 0.0
        # determinism (re-run first play)
        if snapshots:
            test_snap = snapshots[0]
            r1 = dacs_time_series(test_snap, a_max=a_max, v_cap=v_cap, dt=DT,
                                  samples_per_t=samples_per_t, corridor_radius=corridor_radius,
                                  topk_ps=topk_ps, seed=seed)
            r2 = dacs_time_series(test_snap, a_max=a_max, v_cap=v_cap, dt=DT,
                                  samples_per_t=samples_per_t, corridor_radius=corridor_radius,
                                  topk_ps=topk_ps, seed=seed)
            qa['deterministic'] = bool(np.allclose(r1['dacs_series'], r2['dacs_series']))
    except Exception as e:
        qa['error'] = str(e)

    qa_path = os.path.join(out_dir, f"game_{game_id}_qa_report.json")
    with open(qa_path, 'w', encoding='utf-8') as f:
        json.dump(qa, f)

    return df_summary, per_play_json_paths


def list_games_quick(root_dir: str, limit: int = 50) -> pd.DataFrame:
    """List distinct game_ids quickly using supplementary_data.csv (lighter to load)."""
    supp = os.path.join(root_dir, 'analytics', 'data', '114239_nfl_competition_files_published_analytics_final', 'supplementary_data.csv')
    if not os.path.exists(supp):
        raise FileNotFoundError("supplementary_data.csv not found")
    df = pd.read_csv(supp, usecols=['game_id','play_id'])
    games = df[['game_id']].drop_duplicates()
    games = games.sort_values('game_id').reset_index(drop=True)
    if limit:
        games = games.head(limit)
    return games


def main():
    parser = argparse.ArgumentParser(description='Compute Defensive Air Control metrics for one game (Analytics track MVP).')
    parser.add_argument('--game_id', type=int, help='Target game_id to process')
    parser.add_argument('--out_dir', type=str, default=os.path.join('analytics', 'outputs', 'dacs'), help='Output directory for JSON/CSV results')
    parser.add_argument('--samples_per_t', type=int, default=SAMPLES_PER_T)
    parser.add_argument('--corridor_radius', type=float, default=CORRIDOR_RADIUS_YDS)
    parser.add_argument('--topk_ps', type=int, default=PLAYER_SHARE_TOPK)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--residual_model_path', type=str, default=None, help='Path to residual reach model joblib')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual reach corrections')
    parser.add_argument('--list_games', action='store_true', help='List some available game_ids and exit')
    args = parser.parse_args()

    root_dir = os.getcwd()

    if args.list_games:
        games = list_games_quick(root_dir)
        print(games.to_string(index=False))
        return

    if args.game_id is None:
        raise SystemExit('Please specify --game_id (use --list_games to browse).')

    df_summary, paths = compute_dacs_for_game(
        root_dir=root_dir,
        game_id=args.game_id,
        out_dir=os.path.join(root_dir, args.out_dir),
        samples_per_t=args.samples_per_t,
        corridor_radius=args.corridor_radius,
        topk_ps=args.topk_ps,
        seed=args.seed,
        residual_model_path=args.residual_model_path,
        use_residual_model=(not args.no_residual),
    )

    print(df_summary.head(10).to_string(index=False))
    print(f"Saved {len(paths)} per-play JSON files under {args.out_dir}")
    print(f"Saved game summary CSV with {len(df_summary)} plays")


if __name__ == '__main__':
    main()
