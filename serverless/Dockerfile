# Model is cached on the RunPod Network Volume at first worker start.
# This base includes CUDA 12.1 and PyTorch 2.5.1, satisfying VoxCPM2's
# supported runtime without installing a second multi-gigabyte CUDA stack.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates ffmpeg libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -c "import torch, torchaudio, torchcodec; print(f'torch={torch.__version__} torchaudio={torchaudio.__version__} torchcodec={torchcodec.__version__}')"

# Handler code
COPY handler.py .

# Config
ENV VOXCPM_TIMESTEPS=20
ENV RUNPOD_INIT_TIMEOUT=600
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
