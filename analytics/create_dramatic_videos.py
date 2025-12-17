"""
Create dramatic videos (GIFs) of the best DACS plays
- Find longest plays with most DACS action
- Create high-quality visualizations with all players shown
- Focus on plays with good storytelling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse, Rectangle, Wedge
from matplotlib import patches
from pathlib import Path
import json
import math

# Field constants
FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

# Colors
COLOR_DEFENSE = "#d32f2f"
COLOR_OFFENSE = "#1976d2"
COLOR_QB = "#fbc02d"
COLOR_WR = "#0288d1"
COLOR_FIELD = "#2e7d32"
COLOR_GRID = "#ffffff"

DT = 0.1  # Time step (seconds)

def find_best_plays():
    """Find the most dramatic plays for videos"""

    dacs_dir = Path('analytics/outputs/dacs_final_full')
    play_files = list(dacs_dir.glob('**/game_*_play_*.json'))

    # Load supplementary for outcomes
    supp = pd.read_csv('analytics/data/supplementary_data.csv', low_memory=False)
    supp = supp[['game_id', 'play_id', 'pass_result']].drop_duplicates()

    outcome_map = {
        'C': 'Completion', 'Complete': 'Completion',
        'I': 'Incomplete', 'Incomplete': 'Incomplete',
        'IN': 'Interception', 'Interception': 'Interception'
    }

    candidates = {
        'high_dacs_interception': [],
        'low_dacs_completion': [],
        'high_dacs_incomplete': []
    }

    print(f"Analyzing {len(play_files)} plays...")

    for i, pf in enumerate(play_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(play_files)}...")

        with open(pf, 'r') as f:
            data = json.load(f)

        game_id = data['game_id']
        play_id = data['play_id']

        # Get outcome
        outcome_row = supp[(supp['game_id'] == game_id) & (supp['play_id'] == play_id)]
        if outcome_row.empty:
            continue

        outcome = outcome_map.get(outcome_row['pass_result'].iloc[0], None)
        if not outcome:
            continue

        dacs_series = data.get('dacs_series', [])
        if not dacs_series or len(dacs_series) < 10:  # Need at least 10 frames (1 second)
            continue

        max_dacs = max(dacs_series)
        avg_dacs = np.mean(dacs_series)
        final_dacs = dacs_series[-1]
        num_frames = len(dacs_series)

        play_info = {
            'game_id': game_id,
            'play_id': play_id,
            'file': pf,
            'max_dacs': max_dacs,
            'avg_dacs': avg_dacs,
            'final_dacs': final_dacs,
            'num_frames': num_frames,
            'dacs_range': max_dacs - min(dacs_series),
            'score': 0
        }

        # High DACS Interception: High final DACS, long play
        if outcome == 'Interception' and final_dacs > 40:
            play_info['score'] = final_dacs * 0.6 + num_frames * 0.4
            candidates['high_dacs_interception'].append(play_info)

        # Low DACS Completion: Low average DACS, long play
        if outcome == 'Completion' and avg_dacs < 10:
            play_info['score'] = (20 - avg_dacs) * 0.6 + num_frames * 0.4
            candidates['low_dacs_completion'].append(play_info)

        # High DACS Incomplete: High final DACS, long play, big range
        if outcome == 'Incomplete' and final_dacs > 30:
            play_info['score'] = final_dacs * 0.4 + num_frames * 0.3 + play_info['dacs_range'] * 0.3
            candidates['high_dacs_incomplete'].append(play_info)

    # Sort and select best
    results = {}
    for category, plays in candidates.items():
        if plays:
            sorted_plays = sorted(plays, key=lambda x: x['score'], reverse=True)
            results[category] = sorted_plays[0]
            print(f"\n{category}:")
            print(f"  Game: {results[category]['game_id']}, Play: {results[category]['play_id']}")
            print(f"  Frames: {results[category]['num_frames']}, Max DACS: {results[category]['max_dacs']:.1f}")

    return results

def normalize_direction(play_direction, x, y):
    """Normalize coordinates based on play direction"""
    if str(play_direction).lower().startswith("left"):
        return 120.0 - float(x), float(y)
    return float(x), float(y)

def load_tracking_data(game_id, play_id, root_dir='analytics'):
    """Load tracking data for a specific play"""
    tracking_files = [
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w01.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w02.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w03.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w04.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w05.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w06.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w07.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w08.csv',
        Path(root_dir) / 'data' / 'raw' / '114239_nfl_competition_files_published_analytics_final' / 'train' / 'input_2023_w09.csv',
    ]

    for tf in tracking_files:
        if not tf.exists():
            continue
        for chunk in pd.read_csv(tf, chunksize=100000):
            sel = chunk[(chunk['game_id'] == game_id) & (chunk['play_id'] == play_id)]
            if not sel.empty:
                return sel
    return pd.DataFrame()

def create_video(play_file, output_name, output_dir='analytics/outputs/presentation', max_frames=None):
    """Create a high-quality GIF video of a play with all players visualized

    Args:
        play_file: Path to play JSON file
        output_name: Output filename
        output_dir: Output directory
        max_frames: Optional maximum number of frames to render (truncate if longer)
    """

    with open(play_file, 'r') as f:
        data = json.load(f)

    game_id = data['game_id']
    play_id = data['play_id']

    print(f"  Loading tracking data for game {game_id} play {play_id}...")
    df = load_tracking_data(game_id, play_id)

    if df.empty:
        print(f"  Warning: No tracking data found for game {game_id} play {play_id}")
        return

    # Get snap frame and play direction
    frame0 = int(df['frame_id'].min())
    snap = df[df['frame_id'] == frame0].copy()
    play_direction = str(snap['play_direction'].iloc[0]) if 'play_direction' in snap.columns else 'right'

    dacs_series = data.get('dacs_series', [])
    num_frames = len(dacs_series)

    # Apply max_frames limit if specified
    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        dacs_series = dacs_series[:max_frames]
        print(f"  Limiting to {num_frames} frames (requested)")

    # Get ball position from corridor centers (this tracks the ball/catch zone)
    corridor_centers = data.get('corridor_centers', [])
    if corridor_centers:
        ball_positions = np.array(corridor_centers[:num_frames])  # Truncate to num_frames
    else:
        ball_positions = np.full((num_frames, 2), np.nan)

    # Limit to actual tracking data frames
    max_tracking_frame = int(df['frame_id'].max()) - frame0 + 1

    print(f"  Creating video with {num_frames} frames (tracking data: {max_tracking_frame} frames)...")

    # Build player tracking arrays
    players = {}
    for nfl_id in df['nfl_id'].unique():
        if pd.isna(nfl_id):
            continue
        nfl_id = int(nfl_id)
        player_df = df[df['nfl_id'] == nfl_id].sort_values('frame_id')

        # Get initial info from snap
        snap_row = snap[snap['nfl_id'] == nfl_id]
        if snap_row.empty:
            continue
        snap_row = snap_row.iloc[0]

        role = str(snap_row.get('player_role', ''))
        side = str(snap_row.get('player_side', ''))

        # Assign color
        if role.lower() == 'targeted receiver':
            color = COLOR_WR
        elif role.lower() == 'passer':
            color = COLOR_QB
        elif side.lower() == 'defense':
            color = COLOR_DEFENSE
        else:
            color = COLOR_OFFENSE

        # Build position array (aligned with output frames)
        positions = np.full((num_frames, 2), np.nan)
        last_valid_pos = None
        for _, row in player_df.iterrows():
            k = int(row['frame_id']) - frame0
            if 0 <= k < num_frames:
                x, y = normalize_direction(play_direction, row['x'], row['y'])
                positions[k] = [x, y]
                last_valid_pos = [x, y]

        # For frames beyond tracking data, hold the last position
        if last_valid_pos is not None:
            for k in range(max_tracking_frame, num_frames):
                positions[k] = last_valid_pos

        players[nfl_id] = {
            'positions': positions,
            'color': color,
            'role': role,
            'side': side,
            'name': str(snap_row.get('player_name', f'Player {nfl_id}'))
        }

    # Set up figure with three panels
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a1a')

    # Main field (larger)
    ax_field = fig.add_axes([0.05, 0.25, 0.65, 0.70])

    # DACS meter (top right)
    ax_meter = fig.add_axes([0.75, 0.65, 0.20, 0.25])

    # Timeline (bottom right)
    ax_timeline = fig.add_axes([0.75, 0.25, 0.20, 0.30])

    def draw_field(ax):
        """Draw football field"""
        ax.clear()
        ax.set_facecolor(COLOR_FIELD)
        ax.set_xlim(FIELD_X_MIN, FIELD_X_MAX)
        ax.set_ylim(FIELD_Y_MIN, FIELD_Y_MAX)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Yard lines
        for x in np.arange(10, 111, 5):
            lw = 1.5 if x % 10 == 0 else 0.5
            alpha = 0.5 if x % 10 == 0 else 0.3
            ax.axvline(x, color=COLOR_GRID, lw=lw, alpha=alpha)

            # Numbers
            if x % 10 == 0 and 10 < x < 110:
                num = x if x <= 50 else 120 - x
                ax.text(x - 1, 5, str(num), color=COLOR_GRID, fontsize=10, alpha=0.6, rotation=270, va='center')
                ax.text(x - 1, 53.3 - 5, str(num), color=COLOR_GRID, fontsize=10, alpha=0.6, rotation=90, va='center')

        # Endzones
        ax.add_patch(Rectangle((0, FIELD_Y_MIN), 10, FIELD_Y_MAX - FIELD_Y_MIN, facecolor='#1b5e20', alpha=0.6))
        ax.add_patch(Rectangle((110, FIELD_Y_MIN), 10, FIELD_Y_MAX - FIELD_Y_MIN, facecolor='#1b5e20', alpha=0.6))

    def init():
        return []

    def animate(frame):
        # Get DACS for this frame
        current_dacs = dacs_series[frame] if frame < len(dacs_series) else 0

        # Draw field
        draw_field(ax_field)

        # Draw all players
        for nfl_id, player_data in players.items():
            pos = player_data['positions'][frame]
            if np.any(np.isnan(pos)):
                # Find last valid position
                for j in range(frame - 1, -1, -1):
                    if not np.any(np.isnan(player_data['positions'][j])):
                        pos = player_data['positions'][j]
                        break

            if not np.any(np.isnan(pos)):
                # Size based on role
                if player_data['role'].lower() == 'targeted receiver':
                    size = 80
                    edge_width = 2
                elif player_data['role'].lower() == 'passer':
                    size = 70
                    edge_width = 1.5
                else:
                    size = 50
                    edge_width = 1

                ax_field.scatter(pos[0], pos[1], s=size, color=player_data['color'],
                               edgecolor='white', linewidths=edge_width, zorder=5)

        # Draw the ball
        ball_pos = ball_positions[frame]
        if not np.any(np.isnan(ball_pos)):
            # Draw ball with brown color and distinctive style
            ax_field.scatter(ball_pos[0], ball_pos[1], s=100, color='#8B4513',
                           edgecolor='white', linewidths=2, marker='o', zorder=10,
                           alpha=0.9)
            # Add a subtle glow effect
            ax_field.scatter(ball_pos[0], ball_pos[1], s=150, color='#8B4513',
                           alpha=0.2, marker='o', zorder=9)

        # Title with DACS
        ax_field.text(60, 58, f'DACS: {current_dacs:.1f}%',
                     ha='center', fontsize=28, fontweight='bold',
                     color='white', bbox=dict(boxstyle='round,pad=1.0',
                     facecolor='#e74c3c' if current_dacs > 50 else '#3498db',
                     alpha=0.9))

        # Add note if beyond tracking data (ball flight projection)
        if frame >= max_tracking_frame:
            ax_field.text(60, 3, 'Ball Flight Projection',
                         ha='center', fontsize=12, style='italic',
                         color='white', alpha=0.7)

        # DACS Meter
        ax_meter.clear()
        wedge_bg = Wedge((0.5, 0.3), 0.25, 180, 0,
                        facecolor='#34495e', edgecolor='white',
                        linewidth=2, transform=ax_meter.transAxes)
        ax_meter.add_patch(wedge_bg)

        theta_fill = 180 - (current_dacs / 100 * 180)
        color = '#27ae60' if current_dacs > 60 else '#f39c12' if current_dacs > 30 else '#e74c3c'
        wedge_fill = Wedge((0.5, 0.3), 0.25, 180, theta_fill,
                          facecolor=color, edgecolor='white',
                          linewidth=1, transform=ax_meter.transAxes)
        ax_meter.add_patch(wedge_fill)

        ax_meter.text(0.5, 0.35, f'{current_dacs:.0f}%',
                     ha='center', va='center', fontsize=36,
                     fontweight='bold', color=color,
                     transform=ax_meter.transAxes)
        ax_meter.text(0.2, 0.3, '0%', ha='right', va='center', fontsize=11,
                     color='white', fontweight='bold', transform=ax_meter.transAxes)
        ax_meter.text(0.8, 0.3, '100%', ha='left', va='center', fontsize=11,
                     color='white', fontweight='bold', transform=ax_meter.transAxes)
        ax_meter.set_xlim(0, 1)
        ax_meter.set_ylim(0, 1)
        ax_meter.axis('off')

        # Timeline
        ax_timeline.clear()
        time_data = dacs_series[:frame+1]
        frames_x = list(range(len(time_data)))

        ax_timeline.fill_between(frames_x, 0, time_data, alpha=0.3, color='#3498db')
        ax_timeline.plot(frames_x, time_data, linewidth=2, color='#2c3e50')
        ax_timeline.scatter([frame], [current_dacs], s=120, c='#e74c3c',
                           edgecolors='white', linewidths=2, zorder=10)

        ax_timeline.set_xlim(0, num_frames)
        ax_timeline.set_ylim(0, 100)
        ax_timeline.set_xlabel('Frame', fontsize=11, color='white', fontweight='bold')
        ax_timeline.set_ylabel('DACS %', fontsize=11, color='white', fontweight='bold')
        ax_timeline.tick_params(colors='white', labelsize=9)
        ax_timeline.set_facecolor('#2c3e50')
        ax_timeline.grid(True, alpha=0.2, color='white')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DEFENSE,
                      markersize=8, label='Defense', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_OFFENSE,
                      markersize=8, label='Offense', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_QB,
                      markersize=8, label='QB', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_WR,
                      markersize=9, label='Target WR', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8B4513',
                      markersize=9, label='Ball', linestyle='None', markeredgecolor='white', markeredgewidth=1)
        ]
        ax_timeline.legend(handles=legend_elements, loc='upper left',
                          frameon=True, facecolor='#2c3e50', edgecolor='white',
                          fontsize=9, labelcolor='white')

        return []

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=num_frames, interval=100,
                                  blit=False, repeat=True)

    # Save as MP4 (or GIF if ffmpeg not available)
    output_path = Path(output_dir) / output_name

    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, bitrate=1800, codec='libx264')
        anim.save(str(output_path), writer=writer)
        print(f"  [OK] Saved video to {output_path}")
    except RuntimeError:
        # Fallback to GIF
        gif_path = output_path.with_suffix('.gif')
        anim.save(str(gif_path), writer='pillow', fps=10)
        print(f"  [OK] Saved as GIF to {gif_path} (ffmpeg not available)")

    plt.close()

if __name__ == '__main__':
    print("="*60)
    print("Creating Dramatic DACS Videos")
    print("="*60)

    # Find best plays
    best_plays = find_best_plays()

    print(f"\n{'='*60}")
    print("Creating Videos...")
    print("="*60)

    if 'high_dacs_interception' in best_plays:
        print("\n[1/3] High DACS Interception")
        create_video(best_plays['high_dacs_interception']['file'],
                    'video_high_dacs_interception.mp4')

    if 'low_dacs_completion' in best_plays:
        print("\n[2/3] Low DACS Completion")
        create_video(best_plays['low_dacs_completion']['file'],
                    'video_low_dacs_completion.mp4')

    if 'high_dacs_incomplete' in best_plays:
        print("\n[3/3] High DACS Incomplete")
        create_video(best_plays['high_dacs_incomplete']['file'],
                    'video_high_dacs_incomplete.mp4')

    print(f"\n{'='*60}")
    print("Videos created!")
    print("="*60)
