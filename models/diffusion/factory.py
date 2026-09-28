import os
import platform

from .config import is_flux2_model, is_kandinsky_model, is_sdxl_model, load_diffusion_config


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

    if inpaint_on := str(config.get("backend") or "").lower() in {"sdxl_inpaint", "inpaint"}:
        print("🚀 Using SDXL inpaint backend")
        from .sdxl_inpaint_backend import SDXLInpaintBackend

        _backend = SDXLInpaintBackend(config)
        return _backend

    if is_flux2_model(str(config.get("model") or ""), str(config.get("backend") or "")):
        print("🚀 Using FLUX.2 backend")
        from .flux2_backend import Flux2Backend

        _backend = Flux2Backend(config)
        return _backend

    if is_kandinsky_model(str(config.get("model") or ""), str(config.get("backend") or "")):
        print("🚀 Using Kandinsky 5 I2I backend")
        from .kandinsky_backend import Kandinsky5Backend

        _backend = Kandinsky5Backend(config)
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


def park_image_backend() -> None:
    global _backend
    if _backend is None or getattr(_backend, "_parked", False):
        return
    pipe = getattr(_backend, "pipe", None) or getattr(_backend, "model", None)
    if pipe is None:
        return
    print("   ♻️  Parking diffusion on CPU (no reload from disk)")
    try:
        import torch

        pipe.to("cpu")
        _backend._parked = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"⚠️  Diffusion park skipped ({type(exc).__name__}: {exc})")


def wake_image_backend() -> None:
    global _backend
    if _backend is None or not getattr(_backend, "_parked", False):
        return
    pipe = getattr(_backend, "pipe", None) or getattr(_backend, "model", None)
    if pipe is None:
        return
    print("   🎨 Waking diffusion on CUDA")
    try:
        import torch

        if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe.to("cuda")
        elif torch.cuda.is_available():
            pipe.to("cuda")
        _backend._parked = False
    except Exception as exc:
        print(f"⚠️  Diffusion wake failed ({type(exc).__name__}: {exc})")


def unload_image_backend():
    """Drop FLUX weights after generation so VLM scoring can reload."""
    global _backend
    if _backend is None:
        return
    for attr in ("model", "pipe"):
        if hasattr(_backend, attr):
            setattr(_backend, attr, None)
    _backend = None
