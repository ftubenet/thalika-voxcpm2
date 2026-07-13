"""
Thalika VoxCPM2 — RunPod Serverless Handler

Receives text + base64 reference audio, runs VoxCPM2 inference on GPU,
returns base64 PCM WAV output.

Model loading strategy:
  1. Check RunPod Network Volume (/runpod-volume/voxcpm2-model/) first
  2. If not found, download from HuggingFace and cache to volume
  3. Subsequent cold starts load from volume (~30-60 sec)
"""

import os
import sys
import base64
import tempfile
import time
import shutil

import runpod
import numpy as np
import soundfile as sf

# ============================================================================
# MODEL LOADING — with Network Volume caching
# ============================================================================
VOLUME_MODEL_PATH = "/runpod-volume/voxcpm2-model"
FALLBACK_MODEL_PATH = "/tmp/voxcpm2-model"
HF_MODEL_ID = "openbmb/VoxCPM2"
TIMESTEPS = int(os.environ.get("VOXCPM_TIMESTEPS", "20"))

def get_model_path():
    """Resolve model path: network volume > local cache > download."""
    # Check network volume first
    if os.path.isdir(VOLUME_MODEL_PATH) and os.listdir(VOLUME_MODEL_PATH):
        print(f"[thalika] ✅ Model found on network volume: {VOLUME_MODEL_PATH}", flush=True)
        return VOLUME_MODEL_PATH

    # Check if volume mount exists (even if empty)
    volume_available = os.path.isdir("/runpod-volume")

    # Download from HuggingFace
    print(f"[thalika] Downloading model from HuggingFace...", flush=True)
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(HF_MODEL_ID)
    print(f"[thalika] Downloaded to: {cache_dir}", flush=True)

    # Cache to network volume if available
    if volume_available:
        print(f"[thalika] Caching model to network volume...", flush=True)
        if os.path.exists(VOLUME_MODEL_PATH):
            shutil.rmtree(VOLUME_MODEL_PATH)
        shutil.copytree(cache_dir, VOLUME_MODEL_PATH)
        print(f"[thalika] ✅ Model cached to {VOLUME_MODEL_PATH}", flush=True)
        return VOLUME_MODEL_PATH

    # No volume — use HF cache directly
    return cache_dir


print("[thalika-serverless] loading VoxCPM2...", flush=True)
start_time = time.time()

try:
    from voxcpm import VoxCPM
    model_path = get_model_path()
    model = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
    load_time = time.time() - start_time
    print(f"[thalika-serverless] ✅ model loaded in {load_time:.1f}s", flush=True)
except Exception as exc:
    print(f"[thalika-serverless] FATAL: {exc}", file=sys.stderr, flush=True)
    raise

# Warm up
print("[thalika-serverless] warming up...", flush=True)
try:
    _ = model.generate(text="။", cfg_value=2.0, inference_timesteps=10, normalize=False)
    print("[thalika-serverless] ✅ warm-up done", flush=True)
except Exception as e:
    print(f"[thalika-serverless] ⚠️ warm-up failed (non-fatal): {e}", flush=True)


# ============================================================================
# HANDLER
# ============================================================================
def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("text", "").strip()
    if not text:
        return {"error": "Missing required field: text"}

    control = job_input.get("control", "").strip()
    if control:
        text = f"({control}) {text}"

    # Reference audio
    ref_path = None
    ref_b64 = job_input.get("reference_audio_base64", "")
    if ref_b64:
        try:
            ref_bytes = base64.b64decode(ref_b64)
            ref_ext = job_input.get("reference_format", "wav")
            ref_path = os.path.join(tempfile.gettempdir(), f"ref.{ref_ext}")
            with open(ref_path, "wb") as f:
                f.write(ref_bytes)
        except Exception as e:
            return {"error": f"Failed to decode reference audio: {str(e)}"}

    # Generation parameters
    cfg = float(job_input.get("cfg_value", 2.0))
    timesteps = int(job_input.get("inference_timesteps", TIMESTEPS))
    normalize = bool(job_input.get("normalize", True))
    denoise = False
    retry_badcase = bool(job_input.get("retry_badcase", True))

    use_prompt_text = bool(job_input.get("use_prompt_text", False))
    prompt_text = job_input.get("prompt_text", "").strip()

    kwargs = {
        "text": text,
        "cfg_value": cfg,
        "inference_timesteps": timesteps,
        "retry_badcase": retry_badcase,
        "normalize": normalize,
        "denoise": denoise,
    }

    if ref_path:
        # Some versions/configs of VoxCPM load as VoxCPM (v1) instead of VoxCPM2,
        # which expects `prompt_wav_path` instead of `reference_wav_path`.
        try:
            has_ref = type(model.tts_model).__name__ == "VoxCPM2Model"
        except Exception:
            has_ref = True # fallback assumption

        if has_ref:
            kwargs["reference_wav_path"] = ref_path
        else:
            kwargs["prompt_wav_path"] = ref_path
            # VoxCPM v1 strictly requires prompt_text if prompt_wav_path is passed
            if not kwargs.get("prompt_text"):
                kwargs["prompt_text"] = prompt_text if prompt_text else ""

        if use_prompt_text and prompt_text:
            kwargs["prompt_wav_path"] = ref_path
            kwargs["prompt_text"] = prompt_text

    # Inference
    try:
        start = time.time()
        wav = model.generate(**kwargs)
        gen_time = time.time() - start
        print(f"[thalika] generated in {gen_time:.1f}s (ts={timesteps})", flush=True)
    except Exception as e:
        return {"error": f"VoxCPM2 generation failed: {str(e)}"}

    # Encode output
    try:
        out_path = os.path.join(tempfile.gettempdir(), "output.wav")
        sf.write(out_path, np.asarray(wav, dtype=np.float32),
                 model.tts_model.sample_rate, subtype="PCM_16")

        with open(out_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "audio_base64": audio_b64,
            "sample_rate": model.tts_model.sample_rate,
            "format": "wav"
        }
    except Exception as e:
        return {"error": f"Failed to encode output: {str(e)}"}
    finally:
        for p in [ref_path, os.path.join(tempfile.gettempdir(), "output.wav")]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ============================================================================
# ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    if "--local" in sys.argv:
        test_job = {
            "id": "local_test",
            "input": {"text": "မင်္ဂလာပါ", "cfg_value": 2.0,
                      "normalize": True, "inference_timesteps": 10}
        }
        result = handler(test_job)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"OK: audio_base64 length = {len(result['audio_base64'])}")
    else:
        runpod.serverless.start({"handler": handler})
