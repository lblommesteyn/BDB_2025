from PIL import Image
import os
import glob

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gif_dir = os.path.join(repo_root, "analytics", "outputs", "presentation", "gifs")
    
    gifs = glob.glob(os.path.join(gif_dir, "*.gif"))
    print(f"Found {len(gifs)} GIFs in {gif_dir}")
    
    for gif_path in gifs:
        try:
            with Image.open(gif_path) as im:
                # Save first frame
                out_path = gif_path.replace('.gif', '.png')
                im.seek(0)
                im.save(out_path)
                print(f"Saved frame to {out_path}")
        except Exception as e:
            print(f"Failed to extract frame from {gif_path}: {e}")

if __name__ == "__main__":
    main()
