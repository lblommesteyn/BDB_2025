"""
Create dramatic videos (not GIFs) of the best DACS plays
- Find longest plays with most DACS action
- Create high-quality MP4 videos
- Focus on plays with good storytelling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from pathlib import Path
import json

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
            'score': 0  # Will calculate
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

def create_video(play_file, output_name, output_dir='analytics/outputs/presentation'):
    """Create a high-quality MP4 video of a play"""

    with open(play_file, 'r') as f:
        data = json.load(f)

    # Load tracking data for this play
    game_id = data['game_id']
    play_id = data['play_id']

    # Find the output file with tracking
    output_file = play_file.parent / f"game_{game_id}_dacs_timeseries.parquet"
    if not output_file.exists():
        # Try finding in parent directory
        output_file = play_file.parent.parent / f"game_{game_id}" / f"game_{game_id}_dacs_timeseries.parquet"

    if not output_file.exists():
        print(f"  Warning: Could not find tracking for game {game_id}")
        return

    ts = pd.read_parquet(output_file)
    play_ts = ts[ts['play_id'] == play_id].sort_values('k')

    if len(play_ts) == 0:
        print(f"  Warning: No timeseries data for play {play_id}")
        return

    dacs_series = data.get('dacs_series', [])
    num_frames = len(dacs_series)

    print(f"  Creating video with {num_frames} frames...")

    # Set up figure
    fig = plt.figure(figsize=(16, 9), facecolor='#1a1a1a')

    # Main field
    ax_field = fig.add_axes([0.05, 0.15, 0.65, 0.75])

    # DACS meter
    ax_meter = fig.add_axes([0.75, 0.60, 0.20, 0.30])

    # Timeline
    ax_timeline = fig.add_axes([0.75, 0.20, 0.20, 0.30])

    def init():
        ax_field.clear()
        ax_meter.clear()
        ax_timeline.clear()
        return []

    def animate(frame):
        ax_field.clear()
        ax_meter.clear()
        ax_timeline.clear()

        # Field setup
        field = Rectangle((0, 0), 120, 53.3, facecolor='#2d5016', edgecolor='white', linewidth=2)
        ax_field.add_patch(field)

        # Yard lines
        for x in range(10, 120, 10):
            ax_field.plot([x, x], [0, 53.3], 'white', linewidth=1, alpha=0.3)

        # Get DACS for this frame
        current_dacs = dacs_series[frame] if frame < len(dacs_series) else 0

        # Title
        ax_field.text(60, 58, f'DACS: {current_dacs:.1f}%',
                     ha='center', fontsize=24, fontweight='bold',
                     color='white', bbox=dict(boxstyle='round,pad=0.8',
                     facecolor='#e74c3c' if current_dacs > 50 else '#3498db',
                     alpha=0.9))

        ax_field.set_xlim(-5, 125)
        ax_field.set_ylim(-5, 60)
        ax_field.set_aspect('equal')
        ax_field.axis('off')

        # DACS Meter
        from matplotlib.patches import Wedge
        wedge_bg = Wedge((0.5, 0.3), 0.25, 180, 0,
                        facecolor='#ecf0f1', edgecolor='#34495e',
                        linewidth=2, transform=ax_meter.transAxes)
        ax_meter.add_patch(wedge_bg)

        theta_fill = 180 - (current_dacs / 100 * 180)
        color = '#27ae60' if current_dacs > 60 else '#f39c12' if current_dacs > 30 else '#e74c3c'
        wedge_fill = Wedge((0.5, 0.3), 0.25, 180, theta_fill,
                          facecolor=color, edgecolor='white',
                          linewidth=1, transform=ax_meter.transAxes)
        ax_meter.add_patch(wedge_fill)

        ax_meter.text(0.5, 0.35, f'{current_dacs:.0f}%',
                     ha='center', va='center', fontsize=32,
                     fontweight='bold', color=color,
                     transform=ax_meter.transAxes)
        ax_meter.set_xlim(0, 1)
        ax_meter.set_ylim(0, 1)
        ax_meter.axis('off')

        # Timeline
        time_data = dacs_series[:frame+1]
        frames_x = list(range(len(time_data)))

        ax_timeline.fill_between(frames_x, 0, time_data, alpha=0.3, color='#3498db')
        ax_timeline.plot(frames_x, time_data, linewidth=2, color='#2c3e50')
        ax_timeline.scatter([frame], [current_dacs], s=100, c='#e74c3c',
                           edgecolors='white', linewidths=2, zorder=10)

        ax_timeline.set_xlim(0, num_frames)
        ax_timeline.set_ylim(0, 100)
        ax_timeline.set_xlabel('Frame', fontsize=10, color='white')
        ax_timeline.set_ylabel('DACS %', fontsize=10, color='white')
        ax_timeline.tick_params(colors='white', labelsize=8)
        ax_timeline.set_facecolor('#2c3e50')
        ax_timeline.grid(True, alpha=0.2, color='white')

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
