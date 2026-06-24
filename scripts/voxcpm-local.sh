#!/usr/bin/env bash
# Run VoxCPM's own Gradio demo locally. It exposes the SAME /gradio_api/call/generate that the app
# already speaks — so once it's up, just pick "Local" in the app's VoxCPM endpoint and Save & check.
# ponytail: no in-app Python/inference code — local VoxCPM is its own process; the app only swaps URLs.
#
# Needs Python 3.10+ and ideally a GPU (CPU works but is slow). First run downloads the ~2B model.
set -euo pipefail

python3 -m venv .voxcpm-venv
# shellcheck disable=SC1091
source .voxcpm-venv/bin/activate
pip install -U pip voxcpm gradio

# Launch the demo on :7860. The exact entrypoint varies by VoxCPM version — check `python -m voxcpm --help`
# or the repo README (github.com/OpenBMB/VoxCPM). The only invariant the app needs:
#   GET http://localhost:7860/gradio_api/info  ->  a "/generate" endpoint exists.
# If your build's generate signature differs from the public demo's 8 args, append the extra ones via
# VOXCPM2_EXTRA_PARAMS in .env.local (e.g. inference_timesteps, retry_badcase).
echo "Start VoxCPM's Gradio demo on port 7860 (e.g. 'python app.py' in the VoxCPM repo), then verify:"
echo "  curl -s http://localhost:7860/gradio_api/info | head"
