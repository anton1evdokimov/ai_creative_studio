"""Factory for VLM backend (used for product analysis + image scoring + captioning).

On 8GB M1 we intentionally reuse ONE VLM for all 3 tasks to avoid loading 2+ heavy models
into the limited GPU RAM. We default to the smallest working Qwen2-VL variant:
mlx-community/Qwen2-VL-2B-Instruct-4bit  (~2GB GPU RAM, works on 8GB).
"""
import os
import platform

from models.diffusion.config import load_vlm_config


_vlm_instance = None


def _use_mock() -> bool:
    return os.environ.get("AICS_USE_MOCK", "").lower() in {"1", "true", "yes", "on"}


def _use_llm_as_vlm_mock() -> bool:
    """If we have a mock LLM already, we also use it for mock VLM textual answers."""
    return _use_mock()


def create_vlm():
    """Create or return the single shared VLM instance."""
    global _vlm_instance
    if _vlm_instance is not None:
        return _vlm_instance

    if _use_mock():
        from .mock_vlm_backend import MockVLMBackend
        _vlm_instance = MockVLMBackend()
        return _vlm_instance

    # Real MLX VLM path for Apple Silicon
    if platform.system() == "Darwin":
        cfg = load_vlm_config()
        vlm_model = cfg.get("mlx_vlm_model") or "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        print(f"👁️  Using MLX VLM backend: {vlm_model}")
        from .mlx_vlm_backend import MLXVLMBackend
        _vlm_instance = MLXVLMBackend(
            vlm_model,
            max_tokens=int(cfg.get("max_tokens") or 900),
        )
        return _vlm_instance

    raise RuntimeError(
        "No supported VLM backend. Use AICS_USE_MOCK=1 for fast offline testing, "
        "or run on Apple Silicon with mlx-vlm installed."
    )


def unload_vlm():
    """Drop the shared VLM so FLUX can use unified RAM."""
    global _vlm_instance
    if _vlm_instance is None:
        return
    for attr in ("_model", "_processor"):
        if hasattr(_vlm_instance, attr):
            setattr(_vlm_instance, attr, None)
    _vlm_instance = None
