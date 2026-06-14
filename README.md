# ltx2-runpod

RunPod Serverless container running **LTX-2.3** (Lightricks, open-weight, 22B params) —
generates a short video clip **with synchronized audio in a single model call**.
Deployed separately from this repo's `docker-compose.yml`; called by
`workflows/video_gen_workflow.json` via RunPod's HTTP API.

## How this works (read before building)

LTX-2.3 has **no diffusers integration** — the HF model card says diffusers support
is "coming soon". The real inference path is Lightricks' own `ltx-pipelines`
package (github.com/Lightricks/LTX-2), run as a CLI module. The Dockerfile clones
that repo and `uv sync`s its workspace; `handler.py` shells out to
`python -m ltx_pipelines.distilled` (the fastest 2-stage pipeline: 8 distilled
steps, CFG=1, no separate LoRA needed).

### Model weights (~71GB total, downloaded on first cold start)

| File | Repo | Size |
|---|---|---|
| `ltx-2.3-22b-distilled-1.1.safetensors` | `Lightricks/LTX-2.3` | ~46GB |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `Lightricks/LTX-2.3` | ~1GB |
| Gemma-3-12B text encoder (5 shards) | `google/gemma-3-12b-it-qat-q4_0-unquantized` | ~24GB |

`handler.py` downloads these via `huggingface_hub` into `LTX_WEIGHTS_DIR`
(default `/runpod-volume/ltx2-weights`) on first run and reuses them on
subsequent warm/cold starts as long as a **network volume** is attached. The
Gemma repo may require accepting its license on HuggingFace and a valid
`HUGGINGFACE_API_TOKEN` (gated model — see `AI_CONTEXT/KNOWN_PITFALLS.md`).

### GPU / cost requirements (revised)

The original scaffold assumed a 16-24GB GPU / GGUF quantization — **that was
wrong for this model**. There is no GGUF build of LTX-2.3 on HuggingFace.

- **bf16 (default)**: distilled transformer (~46GB) + Gemma-3-12B (~24GB) +
  activations ≈ needs an **80GB GPU (A100/H100)**.
- **fp8** (`LTX_QUANTIZATION=fp8-scaled-mm` or `fp8-cast`): roughly halves the
  transformer's VRAM (~23GB), which may fit a **48GB GPU** (e.g. A6000/L40).
  Quality trade-off untested — try this first if 80GB pricing is too high.
- `LTX_OFFLOAD_MODE=cpu` or `disk` streams weights instead of holding them all
  on GPU — slower, but can unblock smaller cards. Combine with fp8 if needed.

## Build & push

```bash
docker build -t <registry>/<your-namespace>/ltx2-runpod:latest .
docker push <registry>/<your-namespace>/ltx2-runpod:latest
```

Build is slow and disk-heavy: it clones the LTX-2 repo and `uv sync`s a
PyTorch 2.7 / CUDA 12.8 environment. The model weights are **not** baked into
the image — they're pulled into the network volume at runtime.

## Deploy on RunPod Serverless

1. RunPod console → Serverless → New Endpoint.
2. Container image: `<registry>/<your-namespace>/ltx2-runpod:latest`.
3. GPU: **80GB tier (A100/H100)** by default; try a 48GB tier only if you set
   `LTX_QUANTIZATION=fp8-scaled-mm`.
4. **Attach a network volume** (≥100GB) mounted at `/runpod-volume` — this is
   where the ~71GB of weights are cached. Without it, every cold start
   re-downloads ~71GB.
5. Workers: **Min = 0** (scale-to-zero — required for "couple hours/day" cost), Max = 1.
6. Container disk: 30-40GB is enough for the image/venv itself (weights live on
   the network volume).
7. Environment variables: set `HUGGINGFACE_API_TOKEN` if the Gemma repo is gated
   for your account. Other env vars (`LTX_MODEL_REPO`, `LTX_DISTILLED_CHECKPOINT`,
   `LTX_SPATIAL_UPSCALER`, `GEMMA_MODEL_REPO`, `FPS`, `WIDTH`, `HEIGHT`,
   `LTX_QUANTIZATION`, `LTX_OFFLOAD_MODE`) have working defaults baked into the
   Dockerfile — override only if needed.
8. Save, then copy the **Endpoint ID** and create/copy an **API Key**
   (RunPod console → Settings → API Keys).

## Wire into n8n

Add to `.env` (and forwarded via `docker-compose.yml` to `n8n`/`n8n-worker`):

```
RUNPOD_API_KEY=<your runpod api key>
RUNPOD_ENDPOINT_ID=<endpoint id from step above>
```

`workflows/video_gen_workflow.json` calls:

- `POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run`
  body: `{"input": {"prompt": "...", "duration_sec": 4}}`
  header: `Authorization: Bearer $RUNPOD_API_KEY`
  → `{"id": "<job_id>", "status": "IN_QUEUE"}`
- `GET https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/<job_id>`
  → poll until `status` is `COMPLETED` (then `output.video_b64`/`output.error`) or
  `FAILED`.

## Local smoke test (requires an NVIDIA GPU with enough VRAM)

```bash
docker run --rm -it --gpus all \
  -v /path/to/network_volume:/runpod-volume \
  -e HUGGINGFACE_API_TOKEN=<token if needed> \
  <registry>/<your-namespace>/ltx2-runpod:latest \
  /opt/LTX-2/.venv/bin/python3 -c "from handler import handler; import json; \
print(json.dumps(handler({'input': {'prompt': 'a calm ocean at sunset', 'duration_sec': 2}}))[:200])"
```

First run downloads ~71GB of weights into `/runpod-volume` — expect this to
take a long time depending on bandwidth. Decode `output.video_b64` (base64) to
an `.mp4` and run `ffprobe` to confirm both a video and an audio stream are
present.

## Cost notes

RunPod Serverless bills per second of GPU time while a worker is processing a
request, plus per-second idle-keepalive only if `idleTimeout` keeps a worker warm.
With `min workers = 0`, on an 80GB tier (~$1.5-2/hr) a clip taking ~2-4 minutes
costs roughly $0.05-0.15. At "a couple hours/day" total generation time, this
stays in the low single-digit dollars/day range — higher than the original
16-24GB estimate, but still scale-to-zero.

Network volume storage (~100GB) is billed continuously (a few cents/GB/month)
even when no worker is running — this is the trade-off for not re-downloading
~71GB on every cold start.
