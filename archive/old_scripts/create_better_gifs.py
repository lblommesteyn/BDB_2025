"""Create better representative GIFs"""
import sys
sys.path.insert(0, 'analytics')
from create_dramatic_videos import create_video
from pathlib import Path

# 1. Rising DACS Interception - shows defenders converging
play1 = Path('analytics/outputs/dacs_final_full/game_2023091006/game_2023091006_play_431.json')
if play1.exists():
    print("[1/3] Creating Rising DACS Interception...")
    create_video(play1, 'video_high_dacs_interception.mp4')
else:
    print(f"Error: {play1} not found")

# 2. Low DACS Completion - consistently low, shows coverage bust
play2 = Path('analytics/outputs/dacs_final_full/game_2023091100/game_2023091100_play_3167.json')
if play2.exists():
    print("\n[2/3] Creating Low DACS Completion...")
    create_video(play2, 'video_low_dacs_completion.mp4')
else:
    print(f"Error: {play2} not found")

# 3. DACS Battle Incomplete - fluctuating, competitive
play3 = Path('analytics/outputs/dacs_final_full/game_2023091012/game_2023091012_play_2725.json')
if play3.exists():
    print("\n[3/3] Creating DACS Battle Incomplete...")
    create_video(play3, 'video_high_dacs_incomplete.mp4')
else:
    print(f"Error: {play3} not found")

print("\nDone!")
