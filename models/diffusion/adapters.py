"""IP-Adapter Plus (SDXL) — product identity from one reference image.

Weights: h94/IP-Adapter / sdxl_models / ip-adapter-plus_sdxl_vit-h.safetensors
ControlNet is unchanged; this only attaches IP-Adapter on the UNet.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


PLUS_REPO = "h94/IP-Adapter"
PLUS_SUBFOLDER = "sdxl_models"
PLUS_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
PLUS_ENCODER = "models/image_encoder"


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def attach_ip_adapter_plus(pipe, scale: float = 0.6):
    if not hasattr(pipe, "load_ip_adapter"):
        raise RuntimeError("This diffusers pipeline has no load_ip_adapter — upgrade diffusers")
    pipe.load_ip_adapter(
        PLUS_REPO,
        subfolder=PLUS_SUBFOLDER,
        weight_name=PLUS_WEIGHT,
        image_encoder_folder=PLUS_ENCODER,
    )
    pipe.set_ip_adapter_scale(float(scale))
    return pipe
