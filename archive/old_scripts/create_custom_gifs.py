"""Create GIFs with custom frame limits"""
import sys
sys.path.insert(0, 'analytics')
from create_dramatic_videos import create_video
from pathlib import Path

# 1. Interception - stop at frame 6
play1 = Path('analytics/outputs/dacs_final_full/game_2023091006/game_2023091006_play_431.json')
if play1.exists():
    print("[1/2] Creating Interception (stopping at frame 6)...")
    create_video(play1, 'video_high_dacs_interception.mp4', max_frames=6)
else:
    print(f"Error: {play1} not found")

# 2. Completion - stop at frame 24
play2 = Path('analytics/outputs/dacs_final_full/game_2023091100/game_2023091100_play_3167.json')
if play2.exists():
    print("\n[2/2] Creating Completion (stopping at frame 24)...")
    create_video(play2, 'video_low_dacs_completion.mp4', max_frames=24)
else:
    print(f"Error: {play2} not found")

print("\nDone!")
