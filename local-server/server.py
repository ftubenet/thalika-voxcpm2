"""
Thalika local VoxCPM2 inference server.

This is a SEPARATE process from the Next.js app. The app talks to it over HTTP via the
same Gradio /generate endpoint it already uses for the public Hugging Face Space, so no
app code changes are needed — you just point the app at http://localhost:7860.

INVARIANT: the generate() function exposes exactly the same 8 arguments, in the same
order, as the app sends in its Gradio data[] payload
(src/lib/providers/voxcpm2-provider.ts):
    [text, control, audio, use_prompt_text, prompt_text, cfg_value, normalize, denoise]
The api_name MUST be "generate" so the app's POST /gradio_api/call/generate works.

Requirements: Python 3.10-3.12 (3.13+ is not supported by the voxcpm/torch stack).
The ~2B model is downloaded on first run and cached locally.
"""

import os
import sys
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf

from voxcpm import VoxCPM

# ponytail: torch auto-picks the device (CUDA if present, else CPU; MPS support depends on the
# voxcpm build). Add a device knob here only if CPU is too slow and MPS isn't auto-selected.
# VOXCPM_MODEL_DIR points at a pre-downloaded local copy (e.g. from ModelScope); else pull from HF.
MODEL = os.environ.get("VOXCPM_MODEL_DIR", "openbmb/VoxCPM2")
# Denoiser is OFF by default: it adds ~30s startup and the zipenhancer errors on some builds
# (e.g. Apple MPS). Opt in with VOXCPM_LOAD_DENOISER=1 on hardware where it works; the per-request
# `denoise` toggle is a safe no-op while it's off.
LOAD_DENOISER = os.environ.get("VOXCPM_LOAD_DENOISER", "0") not in ("0", "false", "no")
print(f"[thalika-local] loading VoxCPM2 from '{MODEL}' (denoiser={LOAD_DENOISER}) ...", flush=True)
try:
    model = VoxCPM.from_pretrained(MODEL, load_denoiser=LOAD_DENOISER)
    print("[thalika-local] model loaded.", flush=True)
except Exception as exc:  # noqa: BLE001 - surface a clear startup failure
    print(f"[thalika-local] FATAL: could not load VoxCPM2: {exc}", file=sys.stderr, flush=True)
    raise


def generate(text, control, audio, use_prompt_text, prompt_text, cfg_value, normalize, denoise, inference_timesteps=None, retry_badcase=None):
    """Exposes the full VoxCPM2 surface via the app's 8-arg payload, plus local-only controls:
      args 1-8 (app's public-Space signature):
        text, control, audio, use_prompt_text, prompt_text, cfg_value, normalize, denoise
      args 9-10 (local server only — the public Space's 8-arg signature would reject them):
        inference_timesteps : diffusion sampling steps (the biggest quality/stability lever)
        retry_badcase       : auto-retry degenerate takes (repeat/echo) on the server side
      - control non-empty -> "(style/description) text" prefix = Controllable Cloning / Voice Design
      - audio present      -> Controllable Cloning (timbre from the reference)
      - audio absent       -> Voice Design (a brand-new voice from the description, no reference)
      - use_prompt_text + prompt_text -> Ultimate Cloning (exact-transcript continuation)
      - Context-Aware prosody is automatic. (Streaming intentionally not wired — file-based pipeline.)
    """
    if not text or not text.strip():
        raise gr.Error("Text is required.")

    # The model steers style/voice via a "(...)" prefix on the text — that's how control works.
    if control and control.strip():
        text = f"({control.strip()}) {text}"

    # Reference is OPTIONAL: with it -> cloning; without it -> Voice Design.
    ref_path = None
    if audio is not None:
        ref_path = audio if isinstance(audio, str) else None
        if ref_path is None:
            try:
                sample_rate, samples = audio  # (int, ndarray)
            except (TypeError, ValueError) as exc:
                raise gr.Error("Reference audio could not be read.") from exc
            ref_path = os.path.join(tempfile.gettempdir(), "thalika_reference.wav")
            sf.write(ref_path, np.asarray(samples), int(sample_rate))

    cfg = float(cfg_value) if cfg_value is not None else 2.0
    timesteps = int(inference_timesteps) if inference_timesteps else int(os.environ.get("VOXCPM_TIMESTEPS", "10"))
    # retry_badcase: True = auto-retry degenerate (repeat/echo) takes. Defaults True in the model;
    # the app always sends True for local. None (e.g. an older caller) -> keep the model default.
    do_retry = bool(retry_badcase) if retry_badcase is not None else True

    # normalize (text) + denoise (reference) are forwarded; denoise is gated on the denoiser load.
    kwargs = {
        "text": text,
        "cfg_value": cfg,
        "inference_timesteps": timesteps,
        "retry_badcase": do_retry,
        "normalize": bool(normalize),
        "denoise": bool(denoise) and LOAD_DENOISER,
    }
    if ref_path:
        kwargs["reference_wav_path"] = ref_path
        if use_prompt_text and prompt_text and prompt_text.strip():
            kwargs["prompt_wav_path"] = ref_path
            kwargs["prompt_text"] = prompt_text.strip()
    wav = model.generate(**kwargs)

    # Write a 16-bit PCM WAV at the model's real rate (VoxCPM2 = 48kHz). The app requires 48kHz
    # PCM WAV (src/lib/audio-utils.ts) — returning the wrong rate/format breaks its decoder.
    # Must live in the system temp dir (or cwd) or Gradio 6 refuses to serve it. /tmp is NOT it on macOS.
    out_path = os.path.join(tempfile.gettempdir(), "thalika_output.wav")
    sf.write(out_path, np.asarray(wav, dtype=np.float32), model.tts_model.sample_rate, subtype="PCM_16")
    return out_path


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="text"),
        gr.Textbox(label="control"),
        gr.Audio(label="audio", type="filepath"),
        gr.Checkbox(label="use_prompt_text", value=False),
        gr.Textbox(label="prompt_text", value=""),
        gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="cfg_value"),
        gr.Checkbox(label="normalize", value=True),
        gr.Checkbox(label="denoise", value=False),
        gr.Slider(4, 50, value=10, step=1, label="inference_timesteps"),
        gr.Checkbox(label="retry_badcase", value=True),
    ],
    outputs=gr.Audio(label="output"),
    api_name="generate",
    flagging_mode="never",  # Gradio 6 renamed allow_flagging -> flagging_mode
)

if __name__ == "__main__":
    port = int(os.environ.get("VOXCPM_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
