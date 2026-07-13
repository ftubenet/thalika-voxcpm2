# Thalika VoxCPM2 — RunPod Serverless

Deploy VoxCPM2 voice cloning as a serverless endpoint on RunPod.

## What This Does

- **Auto-wake**: Server starts automatically when you generate audio
- **Auto-sleep**: Server stops after ~5 min idle (configurable)  
- **$0 idle cost**: Only pay when generating
- **~30-60 sec cold start**: Model baked into Docker image
- **Per-second billing**: Much cheaper than keeping a GPU pod running

## Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler (VoxCPM2 inference) |
| `download_model.py` | Downloads model during Docker build |
| `Dockerfile` | Docker image definition |
| `requirements.txt` | Python dependencies |
| `test_input.json` | Local testing input |

## Build & Deploy

### 1. Build Docker Image

On a machine with Docker + GPU (or on RunPod itself):

```bash
cd serverless
docker build -t thalika-voxcpm2-serverless .
```

> **Note**: First build downloads ~5GB model. Subsequent builds use cache.

### 2. Push to Docker Hub

```bash
docker tag thalika-voxcpm2-serverless YOUR_DOCKERHUB_USER/thalika-voxcpm2:latest
docker push YOUR_DOCKERHUB_USER/thalika-voxcpm2:latest
```

Or use GitHub Container Registry:
```bash
docker tag thalika-voxcpm2-serverless ghcr.io/YOUR_GITHUB_USER/thalika-voxcpm2:latest
docker push ghcr.io/YOUR_GITHUB_USER/thalika-voxcpm2:latest
```

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Settings:
   - **Container Image**: `YOUR_DOCKERHUB_USER/thalika-voxcpm2:latest`
   - **GPU**: RTX 3090 or RTX A5000 (24GB VRAM minimum)
   - **Active Workers**: 0 (scale to zero)
   - **Max Workers**: 1
   - **Idle Timeout**: 300 (5 minutes)
   - **FlashBoot**: Enable
4. Click **Create**
5. Copy the **Endpoint ID**

### 4. Configure Thalika App

In `.env.local`:

```bash
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
RUNPOD_API_KEY=your_runpod_api_key_here
```

Restart the dev server. The app auto-detects serverless mode.

## Local Testing

```bash
cd serverless
python handler.py --local
```

Or with RunPod's test input:
```bash
python handler.py --test_input test_input.json
```

## API Format

### Input
```json
{
  "input": {
    "text": "Script text to speak",
    "control": "calm expression, natural pacing",
    "reference_audio_base64": "base64 encoded audio",
    "reference_format": "wav",
    "cfg_value": 2.0,
    "normalize": true,
    "denoise": false,
    "inference_timesteps": 20
  }
}
```

### Output
```json
{
  "audio_base64": "base64 encoded PCM WAV",
  "sample_rate": 48000,
  "format": "wav"
}
```

## Cost Estimate

| Usage | Cost |
|-------|------|
| Idle (not generating) | $0 |
| Generating (RTX 3090) | ~$0.00019/sec (~$0.69/hr) |
| 1 hour/day × 30 days | ~$21/month |
| Cold start (first request) | ~30-60 sec |
