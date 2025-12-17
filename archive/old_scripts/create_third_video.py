"""Create the third dramatic video using a play from weeks 1-9"""
import sys
sys.path.insert(0, 'analytics')
from create_dramatic_videos import create_video
from pathlib import Path

# Use the top alternative: Game 2023091013, Play 1687
play_file = Path('analytics/outputs/dacs_final_full/game_2023091013/game_2023091013_play_1687.json')

if play_file.exists():
    print("Creating high DACS incomplete video...")
    create_video(play_file, 'video_high_dacs_incomplete.mp4')
    print("Done!")
else:
    print(f"Error: Could not find {play_file}")
