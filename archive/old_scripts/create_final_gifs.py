"""Create final GIFs with proper frame limits"""
import sys
sys.path.insert(0, 'analytics')
from create_dramatic_videos import create_video
from pathlib import Path

# 1. Interception - 6 frames (already done, keep it)
print("[1/3] Interception already at 6 frames - skipping")

# 2. Completion - keep at 24 frames (should show catch)
# Already at 24, but let's verify
play2 = Path('analytics/outputs/dacs_final_full/game_2023091100/game_2023091100_play_3167.json')
if play2.exists():
    print("\n[2/3] Recreating Completion at 24 frames...")
    create_video(play2, 'video_low_dacs_completion.mp4', max_frames=24)
else:
    print(f"Error: {play2} not found")

# 3. Incomplete - cut at frame 8
play3 = Path('analytics/outputs/dacs_final_full/game_2023102206/game_2023102206_play_3684.json')
if play3.exists():
    print("\n[3/3] Creating Incomplete (stopping at frame 8)...")
    create_video(play3, 'video_high_dacs_incomplete.mp4', max_frames=8)
else:
    print(f"Error: {play3} not found")

print("\nDone!")
