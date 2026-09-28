"""Product keep-mask for SDXL inpaint. White = generate, black = keep packshot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _luma_bg_mask(rgb: Image.Image, white_thr: int = 238) -> Image.Image:
    arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    bg = luma >= int(white_thr)
    # also treat near-uniform light gray as studio sweep
    sat = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    bg |= (luma >= 220) & (sat <= 18)
    return Image.fromarray((bg.astype(np.uint8) * 255), mode="L")


def _rembg_bg_mask(rgb: Image.Image) -> Image.Image | None:
    try:
        from rembg import remove
    except ImportError:
        return None
    cut = remove(rgb.convert("RGBA"))
    if not isinstance(cut, Image.Image):
        cut = Image.fromarray(cut)
    alpha = cut.split()[-1]
    product = np.asarray(alpha, dtype=np.uint8) > 16
    bg = ~product
    return Image.fromarray((bg.astype(np.uint8) * 255), mode="L")


def product_inpaint_mask(
    image: Image.Image,
    *,
    dilate_px: int = 8,
    white_thr: int = 238,
) -> Image.Image:
    """L mask, white = inpaint (background), black = keep product."""
    rgb = image.convert("RGB")
    mask = _rembg_bg_mask(rgb)
    source = "rembg"
    if mask is None:
        mask = _luma_bg_mask(rgb, white_thr=white_thr)
        source = "luma-bg"
    if dilate_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=max(3, dilate_px | 1)))
    print(f"   Inpaint mask: {source}  dilate={dilate_px}px  (white=fill scene)")
    return mask.convert("L")


def letterbox_rgb_mask(
    rgb: Image.Image,
    mask: Image.Image,
    width: int,
    height: int,
    fill=(245, 245, 245),
) -> tuple[Image.Image, Image.Image]:
    rgb = rgb.convert("RGB")
    mask = mask.convert("L")
    w, h = rgb.size
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    rgb_r = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    mask_r = mask.resize((nw, nh), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height), fill)
    mcanvas = Image.new("L", (width, height), 255)  # pads = inpaint
    xy = ((width - nw) // 2, (height - nh) // 2)
    canvas.paste(rgb_r, xy)
    mcanvas.paste(mask_r, xy)
    return canvas, mcanvas


def save_mask(mask: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mask.convert("L").save(path)
