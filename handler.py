"""RunPod Serverless handler for LTX-2.3 (Lightricks) — video + synchronized
audio in one model, run via the `ltx-pipelines` CLI (DistilledPipeline).

LTX-2.3 has no diffusers integration yet (the HF model card says diffusers
support is "coming soon"). The image clones github.com/Lightricks/LTX-2 and
runs `python -m ltx_pipelines.distilled` as a subprocess.

Input (event["input"]):
    prompt:       str   - text prompt describing the shot
    duration_sec: float - target clip length in seconds
    fps:          float - optional, defaults to FPS env var
    width/height: int   - optional, defaults to WIDTH/HEIGHT env vars
                          (final stage-2 resolution, must be divisible by 64)

Output:
    {"video_b64": "<base64 mp4>", "mime": "video/mp4"}
    or {"error": "<message>"} on failure (runpod still reports the job COMPLETED;
    the n8n workflow checks for the "error" key).
"""

import base64
import os
import subprocess
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download

import runpod

LTX_REPO_ROOT = "/opt/LTX-2"
LTX_PYTHON = "/opt/LTX-2/.venv/bin/python3"

LTX_MODEL_REPO = os.environ.get("LTX_MODEL_REPO", "Lightricks/LTX-2.3")
DISTILLED_CHECKPOINT = os.environ.get("LTX_DISTILLED_CHECKPOINT", "ltx-2.3-22b-distilled-1.1.safetensors")
SPATIAL_UPSCALER = os.environ.get("LTX_SPATIAL_UPSCALER", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_MODEL_REPO = os.environ.get("GEMMA_MODEL_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")

WEIGHTS_DIR = os.environ.get("LTX_WEIGHTS_DIR", "/runpod-volume/ltx2-weights")
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

DEFAULT_FPS = float(os.environ.get("FPS", "24"))
DEFAULT_WIDTH = int(os.environ.get("WIDTH", "768"))
DEFAULT_HEIGHT = int(os.environ.get("HEIGHT", "512"))

# Optional: "fp8-cast" or "fp8-scaled-mm" to halve transformer VRAM on smaller GPUs.
LTX_QUANTIZATION = os.environ.get("LTX_QUANTIZATION")
# Optional: "cpu" or "disk" to stream weights instead of keeping them all on GPU.
LTX_OFFLOAD_MODE = os.environ.get("LTX_OFFLOAD_MODE")


def _ensure_weights() -> tuple[str, str, str]:
    """Download (once, cached on the network volume) and return local paths to
    the distilled checkpoint, spatial upsampler, and Gemma text-encoder root."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    checkpoint_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename=DISTILLED_CHECKPOINT,
        local_dir=WEIGHTS_DIR,
        token=HF_TOKEN,
    )
    upscaler_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename=SPATIAL_UPSCALER,
        local_dir=WEIGHTS_DIR,
        token=HF_TOKEN,
    )
    gemma_root = snapshot_download(
        repo_id=GEMMA_MODEL_REPO,
        local_dir=os.path.join(WEIGHTS_DIR, "gemma-3-12b"),
        token=HF_TOKEN,
    )
    return checkpoint_path, upscaler_path, gemma_root


# Downloaded once per worker (cold start); cached on the network volume across runs.
_CHECKPOINT_PATH, _UPSCALER_PATH, _GEMMA_ROOT = _ensure_weights()


def handler(event):
    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    duration_sec = float(job_input.get("duration_sec", 4))
    fps = float(job_input.get("fps", DEFAULT_FPS))
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))

    if not prompt:
        return {"error": "missing required field: prompt"}

    # ltx-pipelines requires num_frames = 8*k + 1 for some non-negative integer k.
    k = max(1, round((duration_sec * fps - 1) / 8))
    num_frames = 8 * k + 1

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "clip.mp4")

        cmd = [
            LTX_PYTHON, "-m", "ltx_pipelines.distilled",
            "--distilled-checkpoint-path", _CHECKPOINT_PATH,
            "--spatial-upsampler-path", _UPSCALER_PATH,
            "--gemma-root", _GEMMA_ROOT,
            "--prompt", prompt,
            "--output-path", out_path,
            "--height", str(height),
            "--width", str(width),
            "--num-frames", str(num_frames),
            "--frame-rate", str(fps),
        ]
        if LTX_QUANTIZATION:
            cmd += ["--quantization", LTX_QUANTIZATION]
        if LTX_OFFLOAD_MODE:
            cmd += ["--offload", LTX_OFFLOAD_MODE]

        result = subprocess.run(cmd, cwd=LTX_REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            return {"error": f"ltx_pipelines.distilled failed: {result.stderr[-4000:]}"}

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"video_b64": video_b64, "mime": "video/mp4"}


runpod.serverless.start({"handler": handler})
