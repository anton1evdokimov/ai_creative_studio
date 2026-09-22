import os
import platform

from models.diffusion.config import load_llm_config


_llm_instance = None


def _use_mock() -> bool:
    return os.environ.get("AICS_USE_MOCK", "").lower() in {"1", "true", "yes", "on"}


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def create_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if _use_mock():
        from .mock_backend import MockLLMBackend
        _llm_instance = MockLLMBackend()
        return _llm_instance

    config = load_llm_config()
    system = platform.system()

    if system == "Darwin":
        print("🧠 Using MLX LLM backend (Apple Silicon)")
        from .mlx_backend import MLXBackend

        _llm_instance = MLXBackend(
            name=config["mlx_model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return _llm_instance

    if _has_cuda():
        print("🧠 Using CUDA LLM backend (NVIDIA)")
        from .cuda_backend import CUDABackend

        _llm_instance = CUDABackend(config["cuda_model"])
        return _llm_instance

    raise RuntimeError(
        "No supported LLM backend found. "
        "Use Apple Silicon (Darwin) with MLX, or NVIDIA with CUDA. "
        "For fast offline testing run with synthetic outputs set:  AICS_USE_MOCK=1 python3.11 main.py"
    )


def unload_llm():
    """Drop the shared LLM so FLUX can use unified RAM."""
    global _llm_instance
    if _llm_instance is None:
        return
    for attr in ("model", "tokenizer"):
        if hasattr(_llm_instance, attr):
            setattr(_llm_instance, attr, None)
    _llm_instance = None