import kagglehub
import os

print("[INFO] Starting VGGFace2 download via kagglehub...")
try:
    # Download latest version
    path = kagglehub.dataset_download("hearfool/vggface2")
    print(f"[SUCCESS] Path to dataset files: {path}")
    
    # Write the path to a file so I can read it later
    with open("vggface2_path.txt", "w") as f:
        f.write(path)
except Exception as e:
    print(f"[ERROR] Failed to download dataset: {e}")
