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

Roadmap:
- VLM product analysis
- Stable Diffusion integration
- CLIP evaluation
- LoRA fine-tuning