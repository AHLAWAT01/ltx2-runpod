# ltx2-runpod

RunPod **Pod** (not Serverless) running **LTX-2.3** (Lightricks, open-weight, 22B params) —
generates a short video clip **with synchronized audio in a single model call**,
served over HTTP. Deployed separately from this repo's `docker-compose.yml`;
called by `workflows/video_gen_workflow.json` via the pod's public proxy URL.

## How this works (read before building)

LTX-2.3 has **no diffusers integration** — the HF model card says diffusers support
is "coming soon". The real inference path is Lightricks' own `ltx-pipelines`
package (github.com/Lightricks/LTX-2), run as a CLI module. The Dockerfile clones
that repo and `uv sync`s its workspace; `app.py` is a small FastAPI server that
shells out to `python -m ltx_pipelines.distilled` (the fastest 2-stage pipeline:
8 distilled steps, CFG=1, no separate LoRA needed) and returns the resulting MP4.

### Why a Pod instead of Serverless

RunPod Serverless workers are ephemeral — without a network volume attached
(which isn't offered for all GPU tiers/regions), the ~71GB of model weights
would re-download on every cold start, costing 20-40+ minutes per request.

A **Pod** is a persistent machine: you attach a network volume (mounted at
`/workspace`), the weights download **once** on first request and persist
across pod stop/start. You start/stop the pod manually (or via RunPod's API)
to control cost — there's no automatic scale-to-zero like Serverless.

### Model weights (~71GB total, downloaded on first request)

| File | Repo | Size |
|---|---|---|
| `ltx-2.3-22b-distilled-1.1.safetensors` | `Lightricks/LTX-2.3` | ~46GB |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `Lightricks/LTX-2.3` | ~1GB |
| Gemma-3-12B text encoder (5 shards) | `google/gemma-3-12b-it-qat-q4_0-unquantized` | ~24GB |

`app.py` downloads these via `huggingface_hub` into `LTX_WEIGHTS_DIR`
(default `/workspace/ltx2-weights`) on the first `/generate` request and
reuses them on subsequent requests and pod restarts, as long as a **network
volume is mounted at `/workspace`**. The Gemma repo is gated — you must
accept its license on HuggingFace and provide a valid `HUGGINGFACE_API_TOKEN`
(see `AI_CONTEXT/KNOWN_PITFALLS.md`).

### GPU / cost requirements

- **bf16 (default)**: distilled transformer (~46GB) + Gemma-3-12B (~24GB) +
  activations ≈ needs an **80GB GPU (A100/H100)**.
- **fp8** (`LTX_QUANTIZATION=fp8-scaled-mm` or `fp8-cast`): roughly halves the
  transformer's VRAM (~23GB), which may fit a **48GB GPU** (e.g. A6000/L40).
  Quality trade-off untested — try this first if 80GB pricing is too high.
- `LTX_OFFLOAD_MODE=cpu` or `disk` streams weights instead of holding them all
  on GPU — slower, but can unblock smaller cards. Combine with fp8 if needed.

## Build & push

This repo's `.github/workflows/build.yml` builds the image via GitHub Actions
(local Docker builds on Apple Silicon require slow QEMU emulation for the
`linux/amd64` target) and pushes to Docker Hub:

```
<your-dockerhub-username>/ltx2-runpod:latest
```

Trigger it from the GitHub repo's **Actions** tab → **Run workflow**.

## Deploy on RunPod (Pod)

1. RunPod console → **Pods** → **Deploy**.
2. Template / Container image: `<your-dockerhub-username>/ltx2-runpod:latest`.
3. GPU: **80GB tier (A100/H100)** by default; try a 48GB tier only if you set
   `LTX_QUANTIZATION=fp8-scaled-mm`.
4. **Attach a network volume** (≥100GB) — mounted at `/workspace`. This is
   where the ~71GB of weights are cached. Without it, every pod restart
   re-downloads ~71GB.
5. Container disk: 30-40GB is enough for the image/venv itself (weights live on
   the network volume).
6. **Expose HTTP port 8000** (RunPod's "Expose HTTP Ports" field) — this gives
   you a public proxy URL like `https://<pod-id>-8000.proxy.runpod.net`.
7. Environment variables: set `HUGGINGFACE_API_TOKEN` (Gemma repo is gated —
   accept its license on HuggingFace first). Other env vars
   (`LTX_MODEL_REPO`, `LTX_DISTILLED_CHECKPOINT`, `LTX_SPATIAL_UPSCALER`,
   `GEMMA_MODEL_REPO`, `FPS`, `WIDTH`, `HEIGHT`, `LTX_QUANTIZATION`,
   `LTX_OFFLOAD_MODE`) have working defaults baked into the Dockerfile —
   override only if needed.
8. Deploy. Once running, copy the proxy URL for port 8000 from the pod's
   **Connect** menu.

## Wire into n8n

Add to `.env` (and forwarded via `docker-compose.yml` to `n8n`/`n8n-worker`):

```
LTX2_POD_URL=https://<pod-id>-8000.proxy.runpod.net
```

`workflows/video_gen_workflow.json` calls:

- `POST $LTX2_POD_URL/generate`
  body: `{"prompt": "...", "duration_sec": 4}`
  → `{"video_b64": "...", "mime": "video/mp4"}` or `{"error": "..."}`

The request blocks until the clip is generated (a few minutes), so the HTTP
Request node's timeout is set generously (10 minutes).

## Smoke test

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a calm ocean at sunset", "duration_sec": 2}' \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('out.mp4','wb').write(base64.b64decode(d['video_b64']))"
```

First call downloads ~71GB of weights into `/workspace` — expect this to take
a long time depending on bandwidth. Subsequent calls (and after pod
stop/start) skip the download. Run `ffprobe out.mp4` to confirm both a video
and an audio stream are present.

## Cost notes

Unlike Serverless, a Pod bills continuously while running (~$1.40/hr for an
80GB A100, similar to the build/test pod used earlier), regardless of whether
a request is in flight. **Stop the pod manually** when not generating videos
to avoid idle billing — only the attached network volume (a few cents/GB/month)
is billed while stopped. Starting the pod again reuses the cached weights on
the network volume, so no re-download is needed.
