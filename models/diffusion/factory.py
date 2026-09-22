import os
import platform

from .config import load_diffusion_config, is_sdxl_model


_backend = None


def _use_mock() -> bool:
    return os.environ.get("AICS_USE_MOCK", "").lower() in {"1", "true", "yes", "on"}


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def create_image_backend():
    global _backend

    if _backend is not None:
        return _backend

    config = load_diffusion_config()

    if _use_mock():
        from .mock_backend import MockImageBackend
        _backend = MockImageBackend(config)
        return _backend

    if is_sdxl_model(config.get("model", "")):
        print("🚀 Using SDXL Diffusers backend")
        from .sdxl_backend import SDXLBackend

        _backend = SDXLBackend(config)
        return _backend

    system = platform.system()

    if system == "Darwin":
        print("🚀 Using FLUX.1 MLX backend (Apple Silicon / mflux)")
        from .mlx_backend import FluxMLXBackend

        _backend = FluxMLXBackend(config)
        return _backend

    if _has_cuda():
        print("🚀 Using FLUX.1 CUDA backend (NVIDIA / Diffusers)")
        from .cuda_backend import FluxCUDABackend

        _backend = FluxCUDABackend(config)
        return _backend

    raise RuntimeError(
        "No supported FLUX.1 backend found. "
        "Use Apple Silicon with mflux, or NVIDIA with Diffusers. "
        "For fast offline testing with synthetic placeholder images run:  AICS_USE_MOCK=1 python3.11 main.py"
    )


def unload_image_backend():
    """Drop FLUX weights after generation so VLM scoring can reload."""
    global _backend
    if _backend is None:
        return
    for attr in ("model", "pipe"):
        if hasattr(_backend, attr):
            setattr(_backend, attr, None)
    _backend = None
