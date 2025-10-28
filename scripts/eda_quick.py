import os, glob, sys, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

np.random.seed(0)

ROOT = os.getcwd()

paths = {
    'analytics_base': os.path.join(ROOT, 'analytics', 'data', '114239_nfl_competition_files_published_analytics_final'),
    'prediction_base': os.path.join(ROOT, 'prediction', 'data'),
}

summary = {}

usecols_in = ['game_id','play_id','player_to_predict','nfl_id','frame_id','play_direction','absolute_yardline_number','player_name','player_height','player_weight','player_birth_date','player_position','player_side','player_role','x','y','s','a','dir','o','num_frames_output','ball_land_x','ball_land_y']

dtypes_in = {
    'game_id': 'int64','play_id': 'int64','player_to_predict':'boolean','nfl_id':'int64','frame_id':'int16',
    'play_direction':'category','absolute_yardline_number':'float32','player_name':'object','player_height':'object',
    'player_weight':'float32','player_birth_date':'object','player_position':'category','player_side':'category','player_role':'category',
    'x':'float32','y':'float32','s':'float32','a':'float32','dir':'float32','o':'float32','num_frames_output':'int16','ball_land_x':'float32','ball_land_y':'float32'
}

usecols_out = ['game_id','play_id','nfl_id','frame_id','x','y']
dtypes_out = {'game_id':'int64','play_id':'int64','nfl_id':'int64','frame_id':'int16','x':'float32','y':'float32'}


def eda_inputs(file_list, label):
    stats = {
        'rows': 0,
        'unique_plays': set(),
        'unique_players': set(),
        'frame_id_min': None,
        'frame_id_max': None,
        'x_min': None, 'x_max': None,
        'y_min': None, 'y_max': None,
        's_sample': [],
        'a_sample': [],
        'num_frames_output_counter': Counter(),
        'player_to_predict_counter': Counter(),
        'play_direction_counter': Counter(),
        'position_counter': Counter(),
        'role_counter': Counter(),
        'side_counter': Counter(),
        'missing': defaultdict(int),
        'ball_land_x_min': None, 'ball_land_x_max': None,
        'ball_land_y_min': None, 'ball_land_y_max': None,
        'files': len(file_list)
    }
    for f in file_list:
        try:
            for chunk in pd.read_csv(f, usecols=usecols_in, dtype=dtypes_in, chunksize=200_000):
                n = len(chunk)
                stats['rows'] += n
                # uniques
                stats['unique_plays'].update(zip(chunk['game_id'].to_numpy(), chunk['play_id'].to_numpy()))
                stats['unique_players'].update(chunk['nfl_id'].unique().tolist())
                # ranges
                cmin = chunk['frame_id'].min(); cmax = chunk['frame_id'].max()
                stats['frame_id_min'] = cmin if stats['frame_id_min'] is None else min(stats['frame_id_min'], cmin)
                stats['frame_id_max'] = cmax if stats['frame_id_max'] is None else max(stats['frame_id_max'], cmax)
                for col, lo_key, hi_key in [('x','x_min','x_max'),('y','y_min','y_max')]:
                    cmi, cma = chunk[col].min(skipna=True), chunk[col].max(skipna=True)
                    stats[lo_key] = cmi if stats[lo_key] is None else min(stats[lo_key], cmi)
                    stats[hi_key] = cma if stats[hi_key] is None else max(stats[hi_key], cma)
                for col, lo_key, hi_key in [('ball_land_x','ball_land_x_min','ball_land_x_max'),('ball_land_y','ball_land_y_min','ball_land_y_max')]:
                    if col in chunk.columns:
                        cmi, cma = chunk[col].min(skipna=True), chunk[col].max(skipna=True)
                        stats[lo_key] = cmi if stats[lo_key] is None else min(stats[lo_key], cmi)
                        stats[hi_key] = cma if stats[hi_key] is None else max(stats[hi_key], cma)
                # samples for s and a
                for col, key in [('s','s_sample'),('a','a_sample')]:
                    arr = chunk[col].dropna().to_numpy()
                    if arr.size:
                        k = min(5000, arr.size)
                        idx = np.random.choice(arr.size, k, replace=False)
                        stats[key].append(arr[idx])
                # counters
                stats['num_frames_output_counter'].update(chunk['num_frames_output'].value_counts(dropna=False).to_dict())
                stats['player_to_predict_counter'].update(chunk['player_to_predict'].value_counts(dropna=False).to_dict())
                stats['play_direction_counter'].update(chunk['play_direction'].value_counts(dropna=False).to_dict())
                stats['position_counter'].update(chunk['player_position'].value_counts(dropna=False).to_dict())
                stats['role_counter'].update(chunk['player_role'].value_counts(dropna=False).to_dict())
                stats['side_counter'].update(chunk['player_side'].value_counts(dropna=False).to_dict())
                # missingness
                for col in usecols_in:
                    if col in chunk.columns:
                        stats['missing'][col] += int(chunk[col].isna().sum())
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
    # finalize
    s_vals = np.concatenate(stats['s_sample']) if stats['s_sample'] else np.array([])
    a_vals = np.concatenate(stats['a_sample']) if stats['a_sample'] else np.array([])
    def q(arr):
        if arr.size == 0:
            return None
        qs = np.percentile(arr, [50,75,90,95,99])
        return {'p50': float(qs[0]), 'p75': float(qs[1]), 'p90': float(qs[2]), 'p95': float(qs[3]), 'p99': float(qs[4])}
    out = {
        'rows': stats['rows'],
        'unique_plays': len(stats['unique_plays']),
        'unique_players': len(stats['unique_players']),
        'frame_id_range': [None if stats['frame_id_min'] is None else int(stats['frame_id_min']), None if stats['frame_id_max'] is None else int(stats['frame_id_max'])],
        'x_range': [None if stats['x_min'] is None else float(stats['x_min']), None if stats['x_max'] is None else float(stats['x_max'])],
        'y_range': [None if stats['y_min'] is None else float(stats['y_min']), None if stats['y_max'] is None else float(stats['y_max'])],
        'ball_land_x_range': [None if stats['ball_land_x_min'] is None else float(stats['ball_land_x_min']), None if stats['ball_land_x_max'] is None else float(stats['ball_land_x_max'])],
        'ball_land_y_range': [None if stats['ball_land_y_min'] is None else float(stats['ball_land_y_min']), None if stats['ball_land_y_max'] is None else float(stats['ball_land_y_max'])],
        'speed_quantiles': q(s_vals),
        'accel_quantiles': q(a_vals),
        'num_frames_output_top': dict(Counter(stats['num_frames_output_counter']).most_common(10)),
        'num_frames_output_minmax': [min(stats['num_frames_output_counter'].keys()) if stats['num_frames_output_counter'] else None, max(stats['num_frames_output_counter'].keys()) if stats['num_frames_output_counter'] else None],
        'player_to_predict_counts': {str(k): int(v) for k,v in stats['player_to_predict_counter'].items()},
        'play_direction_counts': {str(k): int(v) for k,v in Counter(stats['play_direction_counter']).most_common()},
        'position_top10': {str(k): int(v) for k,v in Counter(stats['position_counter']).most_common(10)},
        'role_top10': {str(k): int(v) for k,v in Counter(stats['role_counter']).most_common(10)},
        'side_counts': {str(k): int(v) for k,v in Counter(stats['side_counter']).most_common()},
        'missing_counts': {k:int(v) for k,v in stats['missing'].items()},
        'files': stats['files']
    }
    return out


def eda_outputs(file_list, label):
    stats = {
        'rows': 0,
        'unique_tracks': Counter(),  # key -> count
        'frame_id_min': None,
        'frame_id_max': None,
        'x_min': None, 'x_max': None,
        'y_min': None, 'y_max': None,
        'files': len(file_list)
    }
    for f in file_list:
        try:
            for chunk in pd.read_csv(f, usecols=usecols_out, dtype=dtypes_out, chunksize=200_000):
                n = len(chunk)
                stats['rows'] += n
                # basic ranges
                cmin, cmax = chunk['frame_id'].min(), chunk['frame_id'].max()
                stats['frame_id_min'] = cmin if stats['frame_id_min'] is None else min(stats['frame_id_min'], cmin)
                stats['frame_id_max'] = cmax if stats['frame_id_max'] is None else max(stats['frame_id_max'], cmax)
                for col, lo_key, hi_key in [('x','x_min','x_max'),('y','y_min','y_max')]:
                    cmi, cma = chunk[col].min(skipna=True), chunk[col].max(skipna=True)
                    stats[lo_key] = cmi if stats[lo_key] is None else min(stats[lo_key], cmi)
                    stats[hi_key] = cma if stats[hi_key] is None else max(stats[hi_key], cma)
                # per-track frame counts
                g = chunk.groupby(['game_id','play_id','nfl_id']).size()
                for key, val in g.items():
                    stats['unique_tracks'][key] += int(val)
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
    # summarize per-track counts
    track_lengths = list(stats['unique_tracks'].values())
    length_counter = Counter(track_lengths)
    out = {
        'rows': stats['rows'],
        'tracks': len(track_lengths),
        'frames_per_track_top10': dict(length_counter.most_common(10)),
        'frames_per_track_minmax': [min(track_lengths) if track_lengths else None, max(track_lengths) if track_lengths else None],
        'frame_id_range': [None if stats['frame_id_min'] is None else int(stats['frame_id_min']), None if stats['frame_id_max'] is None else int(stats['frame_id_max'])],
        'x_range': [None if stats['x_min'] is None else float(stats['x_min']), None if stats['x_max'] is None else float(stats['x_max'])],
        'y_range': [None if stats['y_min'] is None else float(stats['y_min']), None if stats['y_max'] is None else float(stats['y_max'])],
        'files': stats['files']
    }
    return out


def eda_supplementary(csv_path):
    if not os.path.exists(csv_path):
        return None
    usecols = ['game_id','play_id','pass_result','route_of_targeted_receiver','team_coverage_type','team_coverage_man_zone','dropback_type','dropback_distance','play_action','pass_length','pass_location_type']
    dtypes = {
        'game_id':'int64','play_id':'int64','pass_result':'category','route_of_targeted_receiver':'category','team_coverage_type':'category','team_coverage_man_zone':'category','dropback_type':'category','dropback_distance':'float32','play_action':'boolean','pass_length':'float32','pass_location_type':'category'
    }
    try:
        df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)
        out = {
            'rows': len(df),
            'unique_plays': int(df[['game_id','play_id']].drop_duplicates().shape[0]),
            'pass_result_counts': df['pass_result'].value_counts(dropna=False).head(10).to_dict(),
            'route_top10': df['route_of_targeted_receiver'].value_counts(dropna=False).head(10).to_dict(),
            'coverage_type_top10': df['team_coverage_type'].value_counts(dropna=False).head(10).to_dict(),
            'man_zone_counts': df['team_coverage_man_zone'].value_counts(dropna=False).to_dict(),
            'dropback_type_counts': df['dropback_type'].value_counts(dropna=False).to_dict(),
            'dropback_distance_stats': {'min': float(df['dropback_distance'].min()), 'p50': float(df['dropback_distance'].median()), 'p90': float(df['dropback_distance'].quantile(0.9)), 'max': float(df['dropback_distance'].max())},
            'pass_length_stats': {'min': float(df['pass_length'].min()), 'p50': float(df['pass_length'].median()), 'p90': float(df['pass_length'].quantile(0.9)), 'max': float(df['pass_length'].max())},
            'pass_location_type_counts': df['pass_location_type'].value_counts(dropna=False).to_dict(),
        }
        return out
    except Exception as e:
        return {'error': str(e)}


def run():
    # Analytics
    a_base = paths['analytics_base']
    a_inputs = sorted(glob.glob(os.path.join(a_base, 'train', 'input_2023_w*.csv')))
    a_outputs = sorted(glob.glob(os.path.join(a_base, 'train', 'output_2023_w*.csv')))
    a_supp = os.path.join(a_base, 'supplementary_data.csv')
    summary['analytics_input'] = eda_inputs(a_inputs, 'analytics_input')
    summary['analytics_output'] = eda_outputs(a_outputs, 'analytics_output')
    summary['analytics_supplementary'] = eda_supplementary(a_supp)

    # Prediction
    p_base = paths['prediction_base']
    p_inputs = sorted(glob.glob(os.path.join(p_base, 'train', 'input_2023_w*.csv')))
    p_outputs = sorted(glob.glob(os.path.join(p_base, 'train', 'output_2023_w*.csv')))
    p_test_in = os.path.join(p_base, 'test_input.csv')
    p_test = os.path.join(p_base, 'test.csv')
    summary['prediction_input'] = eda_inputs(p_inputs, 'prediction_input')
    summary['prediction_output'] = eda_outputs(p_outputs, 'prediction_output')
    # test input quick stats
    summary['prediction_test_input'] = eda_inputs([p_test_in], 'prediction_test_input') if os.path.exists(p_test_in) else None
    # test ids count
    if os.path.exists(p_test):
        df_t = pd.read_csv(p_test, dtype={'game_id':'int64','play_id':'int64','nfl_id':'int64','frame_id':'int16'})
        summary['prediction_test'] = {
            'rows': int(len(df_t)),
            'unique_tracks': int(df_t[['game_id','play_id','nfl_id']].drop_duplicates().shape[0]),
            'frames_per_track_minmax': [int(df_t.groupby(['game_id','play_id','nfl_id']).size().min()), int(df_t.groupby(['game_id','play_id','nfl_id']).size().max())]
        }
    else:
        summary['prediction_test'] = None

    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    run()
