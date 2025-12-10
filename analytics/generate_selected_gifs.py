import json
import os
import subprocess
import sys

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(repo_root, "analytics", "outputs", "presentation", "selected_plays.json")
    
    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        plays = json.load(f)
        
    print(f"Generating GIFs for {len(plays)} plays...")
    
    viz_script = os.path.join(repo_root, "analytics", "visualize_dacs.py")
    out_dir = os.path.join(repo_root, "analytics", "outputs", "presentation", "gifs")
    
    for p in plays:
        gid = p['game_id']
        pid = p['play_id']
        desc = p['desc']
        print(f"Processing {desc} (Game {gid} Play {pid})...")
        
        # We need to find the JSON file for this play.
        # It should be in analytics/outputs/dacs_final_full/game_{gid}/game_{gid}_play_{pid}.json
        # visualize_dacs.py takes --json_dir. It expects the file to be there.
        # We can pass --json_dir analytics/outputs/dacs_final_full/game_{gid}
        
        json_dir = os.path.join(repo_root, "analytics", "outputs", "dacs_final_full", f"game_{gid}")
        
        cmd = [
            sys.executable, viz_script,
            "--game_id", str(gid),
            "--play_id", str(pid),
            "--json_dir", json_dir,
            "--out_dir", out_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            # Rename for clarity
            src = os.path.join(out_dir, f"game_{gid}_play_{pid}.gif")
            dst = os.path.join(out_dir, f"{desc.replace(' ', '_')}.gif")
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
                print(f"Saved to {dst}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate GIF for {gid}-{pid}: {e}")

if __name__ == "__main__":
    main()
