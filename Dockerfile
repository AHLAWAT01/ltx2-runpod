FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

WORKDIR /app

# Ubuntu 24.04 ships Python 3.12 by default. LTX-2's ltx-pipelines package
# requires Python >=3.12, CUDA >12.7, PyTorch ~2.7 (see github.com/Lightricks/LTX-2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git ffmpeg curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# LTX-2.3 has no diffusers integration yet ("coming soon" on the HF model card).
# The real inference path is Lightricks' own ltx-pipelines package, run as a
# CLI module (python -m ltx_pipelines.distilled ...).
ENV LTX_REPO_REF="main"
RUN git clone --depth 1 --branch ${LTX_REPO_REF} https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && cd /opt/LTX-2 \
    && uv sync --frozen

# Extra deps used by app.py itself (HTTP server, HF downloader for weights).
# uv-managed venvs don't ship a pip binary, so install via `uv pip` instead.
RUN cd /opt/LTX-2 && uv pip install --no-cache fastapi "uvicorn[standard]" huggingface_hub

COPY app.py /app/app.py

# Model assets (Lightricks/LTX-2.3 on HuggingFace). Override at deploy time if
# Lightricks publishes new checkpoint filenames.
ENV LTX_MODEL_REPO="Lightricks/LTX-2.3"
ENV LTX_DISTILLED_CHECKPOINT="ltx-2.3-22b-distilled-1.1.safetensors"
ENV LTX_SPATIAL_UPSCALER="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
ENV GEMMA_MODEL_REPO="google/gemma-3-12b-it-qat-q4_0-unquantized"

# Output video defaults (final, stage-2 resolution; must be divisible by 64).
ENV FPS="24"
ENV WIDTH="768"
ENV HEIGHT="512"

# Weights (~71GB: 46GB distilled transformer + ~1GB upscaler + ~24GB Gemma-3-12B)
# are cached on the pod's network volume (mounted at /workspace by RunPod) so
# they survive pod stop/start and aren't re-downloaded every time.
ENV HF_HOME="/workspace/.cache/huggingface"
ENV LTX_WEIGHTS_DIR="/workspace/ltx2-weights"

EXPOSE 8000
CMD ["/opt/LTX-2/.venv/bin/uvicorn", "app:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]
