"""Create competitive incomplete GIF"""
import sys
sys.path.insert(0, 'analytics')
from create_dramatic_videos import create_video
from pathlib import Path

# Best competitive incomplete: shows battle, not total domination
play = Path('analytics/outputs/dacs_final_full/game_2023102206/game_2023102206_play_3684.json')

if play.exists():
    print("Creating Competitive Incomplete (DACS: 0% -> 25%, avg 28%)...")
    create_video(play, 'video_high_dacs_incomplete.mp4')
    print("Done!")
else:
    print(f"Error: {play} not found")
