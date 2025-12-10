import sys
import os
import traceback
sys.path.append(os.getcwd())

from analytics.dacs_one_game import compute_dacs_for_game

try:
    print("Starting compute_dacs_for_game...")
    df, paths = compute_dacs_for_game(
        root_dir=os.getcwd(),
        game_id=2023090700,
        out_dir='analytics/outputs/debug_repro',
        use_outcome_model=False
    )
    print(f"Success! DF shape: {df.shape}")
except Exception:
    traceback.print_exc()
