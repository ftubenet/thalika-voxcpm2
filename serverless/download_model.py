"""
Download VoxCPM2 model during Docker build.
This bakes the model into the Docker image so cold starts don't need to download it.
"""

import os
import shutil
from huggingface_hub import snapshot_download

MODEL_ID = "openbmb/VoxCPM2"
TARGET_DIR = "/app/model"

print(f"[build] Downloading {MODEL_ID} to {TARGET_DIR}...", flush=True)

# Download to HF cache first, then copy to target
cache_dir = snapshot_download(MODEL_ID)
print(f"[build] Downloaded to cache: {cache_dir}", flush=True)

# Copy to a clean path inside the image
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
shutil.copytree(cache_dir, TARGET_DIR)

print(f"[build] ✅ Model saved to {TARGET_DIR}", flush=True)

# Verify
files = os.listdir(TARGET_DIR)
print(f"[build] Model files: {files}", flush=True)
