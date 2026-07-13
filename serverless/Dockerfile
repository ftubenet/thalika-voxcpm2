# Lightweight Dockerfile — model NOT baked in.
# Model downloads to RunPod Network Volume on first cold start.
# Image size: ~3GB (vs ~8GB with baked model).
# Build: docker build -t thalika-voxcpm2-serverless .

FROM runpod/base:0.6.2-cuda11.8.0

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch cu118 to ensure compatibility with older RunPod NVIDIA drivers (like v12040)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Python deps
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Handler code
COPY handler.py .

# Config
ENV VOXCPM_TIMESTEPS=20
ENV RUNPOD_INIT_TIMEOUT=600
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "handler.py"]
