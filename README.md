# AI Creative Studio

Multimodal agentic pipeline for automated product content generation.

## Architecture

User
 ↓
LangGraph Agent
 ↓
LLM Planner
 ↓
Vision Analysis
 ↓
Diffusion Generation
 ↓
Evaluation

## Current status

Implemented:
- LangGraph orchestration
- Local Qwen inference via MLX
- Structured creative planning
- FLUX.1 image generation (MLX/`mflux` on Mac, Diffusers on NVIDIA)

Roadmap:
- VLM product analysis
- CLIP evaluation
- LoRA fine-tuning

## FLUX.1

Default model is **FLUX.1-schnell** (Apache-2.0, 4 steps). Change `diffusion.model` in `config.yaml` to `dev` for FLUX.1-dev.

On Apple Silicon the first run downloads the quantized model through `mflux`. On NVIDIA GPUs it uses Hugging Face Diffusers (`black-forest-labs/FLUX.1-schnell`).

```bash
pip install mflux
python main.py
```

Images are written to `generated/`.