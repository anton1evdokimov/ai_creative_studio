# AI Creative Studio

LangGraph pipeline: product photo → concepts → diffusion ad frames → ranked scores.

Entry points: `python main.py --image PATH` or `python -m uvicorn serve:app --host 0.0.0.0 --port 8080`. Models and backends are set in `config.yaml`.

---

## Pipeline

```mermaid
flowchart TD
  A[analysis] --> P[planning]
  P --> S[scoring]
  S --> R[refine_prompts]
  R --> G[generation]
  G --> E[evaluation]
  E --> V[video I2V]
  V -->|best >= quality_threshold or no retries| END[END]
  V -->|best < threshold| I[improve_prompt]
  I --> R
```

| Stage | Node | What happens | Model |
| --- | --- | --- | --- |
| 1 | `analysis` | VLM reads the packshot → `ProductAnalysis` JSON (category, OCR lines, colours, …) | **VLM** |
| 2 | `planning` | LLM invents N creative concepts (scene, light, style) | **LLM** (text) |
| 3 | `scoring` | LLM rates each concept; keep `top_k_concepts` | **LLM** |
| 4 | `refine_prompts` | LLM writes diffusion `positive_prompt` / `negative_prompt` | **LLM** |
| 5 | `generation` | Draw frames from refined prompts + product photo | **Diffusion** (not LLM/VLM) |
| 6 | `evaluation` | Metrics on PNGs; pick `best_image` | CLIP, DINO, Tesseract, optional **VLM-judge** |
| 6b | `video_generation` | T2V from the winner’s prompt (K5 Lite 10s) | **Kandinsky5T2V** |
| 7 | `quality_router` | If `best.score < quality_threshold` and retries left → `improve_prompt` then refine again | **LLM** on improve |

On **CUDA**, `pipeline.keep_vlm: true` parks Qwen2.5-VL on CPU between stages (one disk load). On **Mac**, VLM is fully unloaded so the LLM/diffusion can use unified RAM.

---

## Which VLM / LLM / vision model

IDs below are the **current `config.yaml` CUDA / MLX defaults**. Change them there.

| Role | When | CUDA | Apple Silicon (MLX) |
| --- | --- | --- | --- |
| Product analysis | Stage 1 | `Qwen/Qwen2.5-VL-32B-Instruct` | `mlx-community/Qwen2-VL-2B-Instruct-4bit` |
| Concepts, concept scoring, prompt refine, improve | Stages 2–4, retry | `Qwen/Qwen2.5-7B-Instruct` | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| VLM-as-judge | Stage 6 if `ranking.vlm_judge: true` | **same VLM instance** as analysis | same 2B MLX VLM |
| CLIP-T / CLIP-I / aesthetic | Stage 6 | `openai/clip-vit-large-patch14` + LAION aesthetic MLP | same |
| DINO-I | Stage 6 | `facebook/dinov2-small` | same |
| CER | Stage 6 | Tesseract `rus+eng` vs VLM OCR text | same (needs `tesseract`) |

The **LLM never sees the image** (except via the JSON dump from stage 1). The **VLM never writes diffusion prompts** unless you change the graph.

Diffusion (`diffusion.backend`) is independent:

| `backend` | How the product photo is used |
| --- | --- |
| `sdxl_inpaint` | SDXL inpaint: keep product pixels, generate background (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) |
| `sdxl` | SDXL T2I; optional ControlNet + IP-Adapter Plus |
| `kandinsky5` | Kandinsky 5 **img2img** (latent from the photo) |
| `kandinsky5_t2i` / UI «только промпт» | Kandinsky 5 **T2I** Lite SFT (`Kandinsky5T2IPipeline`, 1024², 50 steps, gs 3.5). No packshot; CLIP-I / DINO / CER skipped. |
| `flux2` | FLUX.2 Klein: packshot as **visual tokens** next to the text prompt (not I2I) |

---

## Scores

### Stage 3 — concepts (LLM, 1–10)

The text LLM returns JSON:

- `creativity`, `relevance`, `realizability`, `prompt_quality` each **1.0 … 10.0**
- `avg` — mean of those four (parser / model; used to sort)
- `feedback` — one sentence

Winners = top `pipeline.top_k_concepts` by `avg`. **Not** mixed with CLIP.

### Stage 6 — generated images (0–1, then blend)

Each frame gets several 0…1 numbers, then a **weighted average** over metrics that are present (`agent/nodes.py` `_blend`: `sum(w_i * x_i) / sum(w_i)`). Current `ranking.weights`:

| Key | Weight | Meaning |
| --- | --- | --- |
| `clip_t` | 0.20 | Prompt alignment. CLIP ViT-L/14 cosine **image ↔ short English caption** (product + scene, not the full refine prompt). Scaled as CLIPScore: `clip_t = clip(2.5 * max(cos, 0), 0, 1)`. |
| `clip_i` | 0.10 | Identity vs packshot. Image–image cosine mapped `(cos + 1) / 2`. High if the ad still looks like the studio shot. |
| `dino_i` | 0.15 | DINOv2 CLS cosine vs packshot, same `(cos + 1) / 2`. Punishes a new background more than humans do. |
| `aesthetic` | 0.10 | LAION improved aesthetic predictor on CLIP-L image embed, `raw / 10`. |
| `vlm` | 0.30 | VLM-judge `overall` (only if the image passed the gate and judge ran). |
| `cer` | 0.15 | `cer_score = 1 - CER`. CER = Levenshtein(OCR(gen), reference) / len(reference). Reference = VLM `extracted_text_ocr` or `product_name`. **Printed CER is the error** (0 = match). |
| `heuristics` | 0.10 | File/size/blur checks (`analyze_quality` → `quality_factor`). |

**VLM-judge (inside `vlm`):** JSON 0…1 on `prompt_alignment`, `aesthetic_quality`, `product_accuracy`, `realism`, `brand_fit`.

```
overall = 0.30*prompt_alignment + 0.25*aesthetic_quality + 0.20*product_accuracy
        + 0.15*realism + 0.10*brand_fit
```

It is **gated**: by default only images with `pre-score` (blend without VLM) ≥ `vlm_gate.min_pre_score`, CLIP-T ≥ `min_clip_t`, CLIP-I ≥ `min_clip_i`, at most `top_k` frames. If gated out, `vlm` is omitted from the blend (weights renormalize).

**Router:** `quality_threshold` is compared to **final blended `score` of the best image**, not to CLIP-T alone. `0` disables retries.

---

## Run

```bash
# CLI
python main.py --image data/input/product.jpg

# HTTP UI
python -m uvicorn serve:app --host 0.0.0.0 --port 8080
```

CUDA Docker: see `Dockerfile` / `compose.cuda.yaml`. Outputs: `generated/`. Hugging Face cache: `HF_HOME` (e.g. `/mnt/evo4tb/evdokimov/.hf_cache`).

---

## Backend eval harness

Same packshot + prompt through Kandinsky 5 I2I, FLUX.2 Klein, and SDXL inpaint. One row per backend in `generated/eval_backends/metrics.csv`:

| Column | What it is |
| --- | --- |
| `seconds` | Wall time of **generation only** (includes model load if cold). |
| `peak_vram_gb` | `torch.cuda.max_memory_allocated` after that backend. |
| `clip_t` | CLIPScore-style prompt match (0–1), same formula as stage 6. |
| `clip_i` | Packshot identity `(cos+1)/2`. |
| `dino_i` | DINOv2 identity vs packshot. |
| `cer` | Character error vs filename stem as weak label (harness has no VLM OCR). Lower is better. |
| `error` | Empty if the run succeeded. |

```bash
python scripts/eval_backends.py --image data/input/product.jpg \
  --prompt "kefir bottle on a rustic wooden table, morning light" \
  --out_dir generated/eval_backends
```

Hypothesis to log: K5 wins CLIP-I/CER, Klein wins scene (`clip_t` vs a scenic prompt), inpaint locks packshot pixels.

---

## Video (Kandinsky 5 T2V)

After stills are ranked, the **winner’s diffusion prompt** (not the pixels) goes to **Kandinsky 5.0 T2V Lite** distilled 16-step 10s (`kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s`, Diffusers fallback `…-Diffusers`). `guidance_scale=1.0`, 241 frames @ 24 fps ≈ 10 s. **Text-to-video**, not img2vid. If T2V fails, Ken Burns GIF.

`config.yaml` → `video.backend: kandinsky5_t2v`. Disable: `video.enabled: false`.

---

## LoRA + hold-out

Train SDXL UNet LoRA on **seen** SKUs; CLIP-I vs **unseen** photos in `--holdout_data_dir` (same category, different product). If train CLIP-I rises and hold-out does not, the adapter memorized SKUs, not “follow the product”.

```bash
# put train SKUs in data/lora/train, hold-out SKUs in data/lora/holdout
python scripts/train_sdxl_lora.py \
  --instance_data_dir data/lora/train \
  --holdout_data_dir data/lora/holdout \
  --instance_prompt "a photo of a product bottle" \
  --output_dir lora/ref_follow \
  --max_train_steps 500 --eval_every 100
```

`metrics.csv` column `clip_i_holdout`. Then `diffusion.lora.path: lora/ref_follow` with `enabled: true` (SDXL backend, not Klein).

---

## Async GPU jobs

`POST /jobs` (multipart `image`, `description`) returns `202 {"job_id"}` immediately. One GPU lock in `serve_jobs.py`. Poll `GET /jobs/{id}` until `status` is `done` or `error`. UI on `/` does this automatically. `POST /generate` enqueues the same way (no 20‑minute HTTP hold).

