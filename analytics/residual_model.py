import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Shared constants (mirror dacs_one_game defaults)
FRAMERATE_HZ = 10.0
DT = 1.0 / FRAMERATE_HZ
MAX_STEPS_CAP = 120

DEFAULT_A_MAX = 4.75
DEFAULT_V_CAP = 8.20
DEFAULT_A_LAT_MAX = 6.0

ANALYTICS_SUBDIR = os.path.join('analytics', 'data', '114239_nfl_competition_files_published_analytics_final')

USECOLS_IN = [
    'game_id','play_id','player_to_predict','nfl_id','frame_id','play_direction',
    'player_name','player_height','player_weight','player_birth_date','player_position',
    'player_side','player_role','x','y','s','a','dir','o','num_frames_output',
    'ball_land_x','ball_land_y'
]

DTYPES_IN = {
    'game_id':'int64','play_id':'int64','player_to_predict':'boolean','nfl_id':'int64','frame_id':'int16',
    'play_direction':'category','player_name':'object','player_height':'object','player_weight':'float32',
    'player_birth_date':'object','player_position':'category','player_side':'category','player_role':'category',
    'x':'float32','y':'float32','s':'float32','a':'float32','dir':'float32','o':'float32',
    'num_frames_output':'int16','ball_land_x':'float32','ball_land_y':'float32'
}

OUTPUT_USECOLS = ['game_id','play_id','nfl_id','frame_id','x','y']
OUTPUT_DTYPES = {'game_id':'int64','play_id':'int64','nfl_id':'int64','frame_id':'int16','x':'float32','y':'float32'}

FEATURE_NAMES = [
    't',
    't_ratio',
    'speed',
    'accel',
    'cos_dir',
    'sin_dir',
    'dist_ball',
    'cos_ball',
    'dist_wr',
    'cos_wr',
    'dist_qb',
    'cos_qb',
    'pos_cb',
    'pos_s',
    'pos_lb',
    'pos_db',
]

TARGET_NAMES = ['scale_long', 'scale_lat']


@dataclass
class DefenderSnapshot:
    nfl_id: int
    name: str
    pos: Tuple[float, float]
    speed: float
    accel: float
    dir_deg: float
    position: str
    role: str


@dataclass
class PlaySnapshot:
    game_id: int
    play_id: int
    frame_id: int
    play_direction: str
    qb_id: Optional[int]
    qb_pos: Tuple[float, float]
    wr_id: Optional[int]
    wr_pos: Tuple[float, float]
    defenders: List[DefenderSnapshot]
    ball_land: Tuple[float, float]
    num_frames_output: int
    flipped: bool


class ResidualReachModel:
    """Wrapper around scaler + MLP for residual reach scaling."""

    def __init__(
        self,
        feature_scaler: StandardScaler,
        estimator: MLPRegressor,
        feature_names: Sequence[str],
        target_stats: Dict[str, np.ndarray],
        clip_bounds: Tuple[float, float] = (0.0, 3.0),
    ) -> None:
        self.scaler = feature_scaler
        self.model = estimator
        self.feature_names = list(feature_names)
        self.target_stats = target_stats
        self.clip_bounds = clip_bounds

    def predict_scales(self, features: np.ndarray) -> np.ndarray:
        """Return deterministic scale predictions for long and lateral reach."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        X = self.scaler.transform(features)
        preds = self.model.predict(X)
        return np.clip(preds, self.clip_bounds[0], self.clip_bounds[1])

    def sample_scales(self, features: np.ndarray, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample scale realisations using gaussian residual noise."""
        rng = rng or np.random.default_rng()
        if features.ndim == 1:
            features = features.reshape(1, -1)
        mean = self.predict_scales(features)
        std = self.target_stats.get('resid_std', np.zeros_like(mean))
        std = np.broadcast_to(std, mean.shape)
        draws = rng.normal(loc=np.repeat(mean, n_samples, axis=0), scale=np.repeat(std, n_samples, axis=0))
        return np.clip(draws, self.clip_bounds[0], self.clip_bounds[1])


def analytics_input_files(root_dir: str) -> List[str]:
    base = os.path.join(root_dir, ANALYTICS_SUBDIR)
    return sorted(glob.glob(os.path.join(base, 'train', 'input_2023_w*.csv')))


def analytics_output_files(root_dir: str) -> List[str]:
    base = os.path.join(root_dir, ANALYTICS_SUBDIR)
    return sorted(glob.glob(os.path.join(base, 'train', 'output_2023_w*.csv')))


def load_calibration(root_dir: str) -> Tuple[float, float]:
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


def _flip_xy_if_left(play_direction_val: str, x: float, y: float) -> Tuple[float, float]:
    s = str(play_direction_val).lower() if play_direction_val is not None else ''
    if s.startswith('left'):
        return 120.0 - float(x), float(y)
    return float(x), float(y)


def _flip_heading_if_left(play_direction_val: str, dir_deg: float) -> float:
    s = str(play_direction_val).lower() if play_direction_val is not None else ''
    th = float(dir_deg)
    if s.startswith('left'):
        return (180.0 - th) % 360.0
    return th % 360.0


def solve_min_time_to_distance(d0: float, s0: float, a_max: float, v_cap: float) -> float:
    s0 = max(0.0, float(s0))
    a_max = max(1e-6, float(a_max))
    v_cap = max(1e-6, float(v_cap))
    t_acc = max(0.0, (v_cap - s0) / a_max)
    d_acc = s0 * t_acc + 0.5 * a_max * t_acc * t_acc
    if d0 <= d_acc + 1e-9:
        A = 0.5 * a_max
        B = s0
        C = -d0
        disc = B * B - 4 * A * C
        t = (-B + math.sqrt(max(0.0, disc))) / (2 * A)
        return max(0.0, t)
    d_rem = d0 - d_acc
    return t_acc + d_rem / v_cap


def reachable_radius(t: float, s0: float, a_max: float, v_cap: float) -> float:
    s0 = max(0.0, float(s0))
    t = max(0.0, float(t))
    a_max = max(1e-6, float(a_max))
    v_cap = max(1e-6, float(v_cap))
    t_acc = max(0.0, (v_cap - s0) / a_max)
    if t <= t_acc:
        return s0 * t + 0.5 * a_max * t * t
    d_acc = s0 * t_acc + 0.5 * a_max * t_acc * t_acc
    t_rem = t - t_acc
    return d_acc + v_cap * t_rem


def _position_group_flags(position: str) -> Tuple[int, int, int, int]:
    pos = str(position).upper()
    is_cb = int(pos in {'CB', 'DB', 'CB/LB', 'SCB'})
    is_s = int(pos in {'S', 'FS', 'SS', 'SS/FS', 'NB'})
    is_lb = int('LB' in pos and not is_cb)
    is_db = int(pos in {'CB', 'DB', 'S', 'FS', 'SS', 'NB'})
    return is_cb, is_s, is_lb, is_db


def build_feature_vector(
    t: float,
    total_time: float,
    defender: DefenderSnapshot,
    qb_pos: np.ndarray,
    wr_pos: np.ndarray,
    ball_land: np.ndarray,
) -> np.ndarray:
    heading_rad = math.radians(defender.dir_deg)
    cos_dir = math.cos(heading_rad)
    sin_dir = math.sin(heading_rad)
    def_pos = np.asarray(defender.pos, dtype=float)

    def _vec_and_dist(target: np.ndarray) -> Tuple[np.ndarray, float]:
        vec = np.asarray(target, dtype=float) - def_pos
        dist = float(np.linalg.norm(vec))
        return (vec, dist)

    vec_ball, dist_ball = _vec_and_dist(ball_land)
    vec_wr, dist_wr = _vec_and_dist(wr_pos)
    vec_qb, dist_qb = _vec_and_dist(qb_pos)

    def _cos_to(vec: np.ndarray, dist: float) -> float:
        if dist <= 1e-6:
            return 1.0
        unit_vec = vec / dist
        return float(np.dot(unit_vec, np.array([cos_dir, sin_dir])))

    cos_ball = _cos_to(vec_ball, dist_ball)
    cos_wr = _cos_to(vec_wr, dist_wr)
    cos_qb = _cos_to(vec_qb, dist_qb)

    pos_cb, pos_s, pos_lb, pos_db = _position_group_flags(defender.position)

    feat = np.array([
        float(t),
        float(t / max(total_time, 1e-6)),
        float(defender.speed),
        float(defender.accel),
        float(cos_dir),
        float(sin_dir),
        float(dist_ball),
        float(cos_ball),
        float(dist_wr),
        float(cos_wr),
        float(dist_qb),
        float(cos_qb),
        float(pos_cb),
        float(pos_s),
        float(pos_lb),
        float(pos_db),
    ], dtype=float)
    return feat


def _play_snapshots(df_game: pd.DataFrame) -> Iterable[PlaySnapshot]:
    for play_id, df_play in df_game.groupby('play_id'):
        f0 = int(df_play['frame_id'].min())
        snap = df_play[df_play['frame_id'] == f0].copy()
        if snap.empty:
            continue
        play_dir_val = snap['play_direction'].iloc[0] if 'play_direction' in snap.columns else ''
        raw_blx = float(snap['ball_land_x'].iloc[0])
        raw_bly = float(snap['ball_land_y'].iloc[0])
        blx, bly = _flip_xy_if_left(play_dir_val, raw_blx, raw_bly)

        qb = snap[snap['player_role'] == 'Passer']
        wr = snap[snap['player_role'] == 'Targeted Receiver']
        qb_id = int(qb['nfl_id'].iloc[0]) if not qb.empty else None
        wr_id = int(wr['nfl_id'].iloc[0]) if not wr.empty else None

        if not qb.empty:
            qx, qy = _flip_xy_if_left(play_dir_val, float(qb['x'].iloc[0]), float(qb['y'].iloc[0]))
        else:
            qx, qy = _flip_xy_if_left(play_dir_val, float(snap['x'].mean()), float(snap['y'].mean()))
        if not wr.empty:
            wx, wy = _flip_xy_if_left(play_dir_val, float(wr['x'].iloc[0]), float(wr['y'].iloc[0]))
        else:
            wx, wy = _flip_xy_if_left(play_dir_val, float(snap['x'].mean()), float(snap['y'].mean()))

        defs = []
        d_rows = snap[snap['player_side'] == 'Defense']
        if d_rows.empty:
            continue
        for _, r in d_rows.iterrows():
            dx, dy = _flip_xy_if_left(play_dir_val, float(r['x']), float(r['y']))
            ddir = float(r['dir']) if not pd.isna(r['dir']) else 0.0
            ddir = _flip_heading_if_left(play_dir_val, ddir)
            defs.append(DefenderSnapshot(
                nfl_id=int(r['nfl_id']),
                name=str(r.get('player_name', '')),
                pos=(dx, dy),
                speed=float(r['s']) if not pd.isna(r['s']) else 0.0,
                accel=float(r['a']) if not pd.isna(r.get('a', np.nan)) else 0.0,
                dir_deg=ddir,
                position=str(r.get('player_position', '')),
                role=str(r.get('player_role', '')),
            ))

        T_frames = int(np.clip(int(snap['num_frames_output'].max()), 1, MAX_STEPS_CAP))
        yield PlaySnapshot(
            game_id=int(snap['game_id'].iloc[0]),
            play_id=int(play_id),
            frame_id=f0,
            play_direction=str(play_dir_val),
            qb_id=qb_id,
            qb_pos=(qx, qy),
            wr_id=wr_id,
            wr_pos=(wx, wy),
            defenders=defs,
            ball_land=(blx, bly),
            num_frames_output=T_frames,
            flipped=str(play_dir_val).lower().startswith('left'),
        )


def _group_outputs_by_play(df_outputs: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    grouped: Dict[int, pd.DataFrame] = {}
    if df_outputs.empty:
        return grouped
    for pid, g in df_outputs.groupby('play_id'):
        grouped[int(pid)] = g.copy()
    return grouped


def collect_residual_samples(
    root_dir: str,
    game_ids: Optional[Sequence[int]] = None,
    max_games: Optional[int] = None,
    max_samples_per_play: Optional[int] = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Build residual dataset for defender reach scaling."""
    rng = np.random.default_rng(seed)
    a_max, v_cap = load_calibration(root_dir)
    input_files = analytics_input_files(root_dir)
    output_files = analytics_output_files(root_dir)

    if game_ids is None:
        supp_path = os.path.join(root_dir, ANALYTICS_SUBDIR, 'supplementary_data.csv')
        if os.path.exists(supp_path):
            supp = pd.read_csv(supp_path, usecols=['game_id'])
            game_ids = sorted(supp['game_id'].unique().tolist())
        else:
            # fallback: derive from input files (expensive but rare)
            ids = set()
            for f in input_files:
                for chunk in pd.read_csv(f, usecols=['game_id'], dtype={'game_id':'int64'}, chunksize=200_000):
                    ids.update(chunk['game_id'].unique().tolist())
            game_ids = sorted(ids)
    game_ids = list(game_ids)
    if max_games:
        game_ids = game_ids[:max_games]

    rows: List[Dict[str, float]] = []

    for game_id in game_ids:
        df_game_parts = []
        for f in input_files:
            for chunk in pd.read_csv(f, usecols=USECOLS_IN, dtype=DTYPES_IN, chunksize=200_000):
                g = chunk[chunk['game_id'] == game_id]
                if not g.empty:
                    df_game_parts.append(g)
        if not df_game_parts:
            continue
        df_game = pd.concat(df_game_parts, ignore_index=True)

        df_out_parts = []
        for f in output_files:
            for chunk in pd.read_csv(f, usecols=OUTPUT_USECOLS, dtype=OUTPUT_DTYPES, chunksize=200_000):
                g = chunk[chunk['game_id'] == game_id]
                if not g.empty:
                    df_out_parts.append(g)
        if not df_out_parts:
            continue
        df_outputs = pd.concat(df_out_parts, ignore_index=True)
        outputs_by_play = _group_outputs_by_play(df_outputs)

        for snap in _play_snapshots(df_game):
            play_output = outputs_by_play.get(snap.play_id)
            if play_output is None or play_output.empty:
                continue
            play_rows: List[Dict[str, float]] = []
            if snap.flipped:
                play_output = play_output.copy()
                play_output['x'] = 120.0 - play_output['x']
            total_time = float(snap.num_frames_output) * DT
            ball = np.asarray(snap.ball_land, dtype=float)
            qb = np.asarray(snap.qb_pos, dtype=float)
            wr = np.asarray(snap.wr_pos, dtype=float)

            grouped = {int(k): v for k, v in play_output.groupby('nfl_id')}
            for defender in snap.defenders:
                traj = grouped.get(defender.nfl_id)
                if traj is None or traj.empty:
                    continue
                traj = traj.sort_values('frame_id')
                frames = traj['frame_id'].to_numpy(dtype=int)
                # Clamp to available horizon
                max_frame = min(int(frames.max()), snap.num_frames_output)
                frames = frames[frames <= max_frame]
                if frames.size == 0:
                    continue
                positions = traj.loc[traj['frame_id'].isin(frames), ['x','y']].to_numpy(dtype=float)
                init = np.asarray(defender.pos, dtype=float)
                heading_rad = math.radians(defender.dir_deg)
                cos_h = math.cos(heading_rad)
                sin_h = math.sin(heading_rad)
                rot = np.array([[cos_h, sin_h], [-sin_h, cos_h]])

                for frame in frames:
                    if frame > snap.num_frames_output:
                        break
                    idx = np.where(frames == frame)[0]
                    if idx.size == 0:
                        continue
                    pos = positions[idx[0]]
                    delta = pos - init
                    local = rot @ delta
                    forward = max(0.0, float(local[0]))
                    lateral = abs(float(local[1]))
                    t = frame * DT
                    physics_a = reachable_radius(t, defender.speed, a_max, v_cap)
                    b_limit = (defender.speed * defender.speed) / max(DEFAULT_A_LAT_MAX, 1e-6)
                    physics_b = min(physics_a, b_limit)
                    scale_long = forward / max(physics_a, 1e-6)
                    scale_lat = lateral / max(physics_b, 1e-6) if physics_b > 1e-6 else 0.0
                    scale_long = float(np.clip(scale_long, 0.0, 3.0))
                    scale_lat = float(np.clip(scale_lat, 0.0, 3.0))

                    feat = build_feature_vector(
                        t=t,
                        total_time=total_time,
                        defender=defender,
                        qb_pos=qb,
                        wr_pos=wr,
                        ball_land=ball,
                    )
                    play_rows.append({
                        'game_id': snap.game_id,
                        'play_id': snap.play_id,
                        'nfl_id': defender.nfl_id,
                        'frame_id': frame,
                        **{name: feat[i] for i, name in enumerate(FEATURE_NAMES)},
                        'scale_long': scale_long,
                        'scale_lat': scale_lat,
                    })

            if max_samples_per_play and play_rows and len(play_rows) > max_samples_per_play:
                idx = rng.choice(len(play_rows), size=max_samples_per_play, replace=False)
                rows.extend(play_rows[i] for i in idx)
            else:
                rows.extend(play_rows)

    if not rows:
        return pd.DataFrame(columns=['game_id','play_id','nfl_id','frame_id', *FEATURE_NAMES, *TARGET_NAMES])
    return pd.DataFrame(rows)


def train_residual_model(
    df: pd.DataFrame,
    hidden_layer_sizes: Tuple[int, ...] = (32, 32),
    learning_rate_init: float = 0.001,
    max_iter: int = 400,
    random_state: int = 42,
) -> ResidualReachModel:
    if df.empty:
        raise ValueError("Residual dataset is empty; cannot train model.")
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df[TARGET_NAMES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )
    mlp.fit(X_scaled, y)
    preds = mlp.predict(X_scaled)
    resid = y - preds
    resid_std = np.std(resid, axis=0, ddof=1)

    target_stats = {
        'mean': y.mean(axis=0),
        'std': y.std(axis=0, ddof=1),
        'resid_std': resid_std,
    }
    return ResidualReachModel(scaler, mlp, FEATURE_NAMES, target_stats)


def save_residual_model(model: ResidualReachModel, path: str) -> None:
    payload = {
        'feature_names': model.feature_names,
        'scaler': model.scaler,
        'mlp': model.model,
        'target_stats': model.target_stats,
        'clip_bounds': model.clip_bounds,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(payload, path)


def load_residual_model(path: str) -> Optional[ResidualReachModel]:
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    return ResidualReachModel(
        feature_scaler=payload['scaler'],
        estimator=payload['mlp'],
        feature_names=payload['feature_names'],
        target_stats=payload['target_stats'],
        clip_bounds=tuple(payload.get('clip_bounds', (0.0, 3.0))),
    )


def train_cli() -> None:
    parser = argparse.ArgumentParser(description='Train residual reach model for DACS.')
    parser.add_argument('--root_dir', type=str, default='.', help='Repository root directory.')
    parser.add_argument('--game_limit', type=int, default=None, help='Limit number of games for training.')
    parser.add_argument('--samples_per_play', type=int, default=400, help='Max residual samples per play.')
    parser.add_argument('--hidden_layers', type=str, default='32,32', help='Comma-separated hidden layer sizes.')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_path', type=str, default=os.path.join('analytics','models','residual_model.joblib'))
    args = parser.parse_args()

    hidden = tuple(int(x.strip()) for x in args.hidden_layers.split(',') if x.strip())
    df = collect_residual_samples(
        root_dir=args.root_dir,
        game_ids=None,
        max_games=args.game_limit,
        max_samples_per_play=args.samples_per_play,
        seed=args.seed,
    )
    if df.empty:
        raise SystemExit("No residual samples gathered; cannot train.")
    model = train_residual_model(
        df,
        hidden_layer_sizes=hidden,
        learning_rate_init=args.learning_rate,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    out_path = os.path.join(args.root_dir, args.out_path)
    save_residual_model(model, out_path)
    print(f"Saved residual model to {out_path}")
    print(f"Training samples: {len(df)}")
    print(f"Residual std: {model.target_stats.get('resid_std')}")


if __name__ == '__main__':
    train_cli()
