import json, os
fp = 'eda_summary.json'
with open(fp, 'r', encoding='utf-8') as f:
    data = json.load(f)

def rng(x):
    return 'None' if x is None else f"{x[0]} to {x[1]}"

def topn(d, n=5):
    if not d: return '{}'
    return ', '.join([f"{k}:{v}" for k,v in list(d.items())[:n]])

def qpair(qs):
    if not qs: return 'n/a'
    return f"p95={qs.get('p95'):.2f}, p99={qs.get('p99'):.2f}"

def show_inputs(name, s):
    print(f"--- {name} INPUT ---")
    print(f"rows: {s['rows']:,}; unique_plays: {s['unique_plays']:,}; unique_players: {s['unique_players']:,}")
    print(f"frame_id_range: {s['frame_id_range']}")
    print(f"x_range: {s['x_range']}, y_range: {s['y_range']}")
    print(f"ball_land_x_range: {s['ball_land_x_range']}, ball_land_y_range: {s['ball_land_y_range']}")
    print(f"num_frames_output min..max: {s['num_frames_output_minmax']}; top: {topn(s['num_frames_output_top'], 5)}")
    print(f"player_to_predict: {s['player_to_predict_counts']}")
    print(f"play_direction: {s['play_direction_counts']}")
    print(f"positions top: {s['position_top10']}")
    print(f"roles top: {s['role_top10']}")
    print(f"speed: {qpair(s['speed_quantiles'])}; accel: {qpair(s['accel_quantiles'])}")
    miss = {k:v for k,v in s['missing_counts'].items() if v>0}
    if miss:
        print(f"missing>0: {miss}")
    print()

def show_outputs(name, s):
    print(f"--- {name} OUTPUT ---")
    print(f"rows: {s['rows']:,}; tracks: {s['tracks']:,}")
    print(f"frame_id_range: {s['frame_id_range']}")
    print(f"x_range: {s['x_range']}, y_range: {s['y_range']}")
    print(f"frames_per_track min..max: {s['frames_per_track_minmax']}; top: {s['frames_per_track_top10']}")
    print()

# Analytics
show_inputs('Analytics', data['analytics_input'])
show_outputs('Analytics', data['analytics_output'])
print('--- Analytics SUPPLEMENTARY ---')
print({k:data['analytics_supplementary'][k] for k in ['rows','unique_plays']})
print('pass_result:', data['analytics_supplementary']['pass_result_counts'])
print('route_top10:', data['analytics_supplementary']['route_top10'])
print('coverage_type_top10:', data['analytics_supplementary']['coverage_type_top10'])
print('man_zone_counts:', data['analytics_supplementary']['man_zone_counts'])
print('dropback_type_counts:', data['analytics_supplementary']['dropback_type_counts'])
print('dropback_distance_stats:', data['analytics_supplementary']['dropback_distance_stats'])
print('pass_length_stats:', data['analytics_supplementary']['pass_length_stats'])
print('pass_location_type_counts:', data['analytics_supplementary']['pass_location_type_counts'])
print()

# Prediction
show_inputs('Prediction', data['prediction_input'])
show_outputs('Prediction', data['prediction_output'])
if data.get('prediction_test_input'):
    show_inputs('Prediction TEST_INPUT', data['prediction_test_input'])
if data.get('prediction_test'):
    print('--- Prediction TEST ---')
    print(data['prediction_test'])
