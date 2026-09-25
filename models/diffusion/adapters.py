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


def pad_to_square(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    """CLIP ViT-H center-crops to square — pad first so a tall bottle stays whole."""
    image = image.convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def canvas_hw(
    image: Image.Image,
    long_side: int = 1024,
    max_aspect: float = 4 / 3,
) -> tuple[int, int]:
    """SDXL + IP-Adapter break on needle aspect (e.g. 256×1024) and zoom-crop the product.

    Clamp to ≤4:3 so a tall bottle gets 768×1024, not a 1:4 strip.
    """
    w, h = image.size
    long_side = max(64, (int(long_side) // 64) * 64)
    portrait = h >= w
    src = (h / max(w, 1)) if portrait else (w / max(h, 1))
    aspect = min(float(src), float(max_aspect))
    aspect = max(aspect, 1.0)
    short = max(64, (round(long_side / aspect) // 64) * 64)
    if portrait:
        return short, long_side
    return long_side, short


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
