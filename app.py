"""LTX-2.3 (Lightricks) video+audio generation server, run as a persistent
RunPod Pod (not Serverless) so the ~71GB of model weights stay on the pod's
network volume across requests instead of re-downloading on every cold start.

LTX-2.3 has no diffusers integration yet (the HF model card says diffusers
support is "coming soon"). The image clones github.com/Lightricks/LTX-2 and
runs `python -m ltx_pipelines.distilled` as a subprocess.

RunPod's pod proxy is fronted by Cloudflare, which enforces a ~100s timeout
on any single HTTP request — far shorter than a generation run. So this API
is async: submit a job, then poll for its result.

POST /generate
    body: {"prompt": str, "duration_sec": float, "fps"?: float, "width"?: int, "height"?: int}
    response: {"id": "<job_id>", "status": "IN_QUEUE"}

GET /status/{job_id}
    response: {"status": "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED",
                "output"?: {"video_b64": "<base64 mp4>", "mime": "video/mp4"}
                            or {"error": "<message>"}}

GET /health
    response: {"status": "ok"}
"""

import base64
import os
import subprocess
import tempfile
import threading
import uuid

from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel

LTX_REPO_ROOT = "/opt/LTX-2"
LTX_PYTHON = "/opt/LTX-2/.venv/bin/python3"

LTX_MODEL_REPO = os.environ.get("LTX_MODEL_REPO", "Lightricks/LTX-2.3")
DISTILLED_CHECKPOINT = os.environ.get("LTX_DISTILLED_CHECKPOINT", "ltx-2.3-22b-distilled-1.1.safetensors")
SPATIAL_UPSCALER = os.environ.get("LTX_SPATIAL_UPSCALER", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_MODEL_REPO = os.environ.get("GEMMA_MODEL_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")

# RunPod Pods mount a network volume at /workspace by default — weights cached
# here persist across pod stop/start (unlike Serverless, which is ephemeral).
WEIGHTS_DIR = os.environ.get("LTX_WEIGHTS_DIR", "/workspace/ltx2-weights")
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


app = FastAPI()

# Downloaded on first job after pod start; cached on the network volume
# (/workspace) so subsequent pod stop/start cycles skip the ~71GB download.
_weights = None

# In-memory job store: job_id -> {"status": ..., "output": ...}
_jobs: dict[str, dict] = {}

# Serializes GPU work — only one ltx_pipelines run at a time on a single GPU.
_gpu_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    duration_sec: float = 4
    fps: float | None = None
    width: int | None = None
    height: int | None = None


def _run_job(job_id: str, req: GenerateRequest):
    global _weights
    _jobs[job_id]["status"] = "IN_PROGRESS"

    try:
        with _gpu_lock:
            if _weights is None:
                _weights = _ensure_weights()
            checkpoint_path, upscaler_path, gemma_root = _weights

            fps = req.fps or DEFAULT_FPS
            width = req.width or DEFAULT_WIDTH
            height = req.height or DEFAULT_HEIGHT

            # ltx-pipelines requires num_frames = 8*k + 1 for some non-negative integer k.
            k = max(1, round((req.duration_sec * fps - 1) / 8))
            num_frames = 8 * k + 1

            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "clip.mp4")

                cmd = [
                    LTX_PYTHON, "-m", "ltx_pipelines.distilled",
                    "--distilled-checkpoint-path", checkpoint_path,
                    "--spatial-upsampler-path", upscaler_path,
                    "--gemma-root", gemma_root,
                    "--prompt", req.prompt,
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
                    _jobs[job_id]["status"] = "FAILED"
                    _jobs[job_id]["output"] = {"error": f"ltx_pipelines.distilled failed: {result.stderr[-4000:]}"}
                    return

                with open(out_path, "rb") as f:
                    video_b64 = base64.b64encode(f.read()).decode("utf-8")

        _jobs[job_id]["status"] = "COMPLETED"
        _jobs[job_id]["output"] = {"video_b64": video_b64, "mime": "video/mp4"}
    except Exception as exc:
        _jobs[job_id]["status"] = "FAILED"
        _jobs[job_id]["output"] = {"error": str(exc)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "IN_QUEUE", "output": None}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"id": job_id, "status": "IN_QUEUE"}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"status": job["status"], "output": job["output"]}
