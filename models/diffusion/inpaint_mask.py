"""Product keep-mask for SDXL inpaint. White = generate, black = keep packshot."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _flood_true(seed_mask: np.ndarray) -> np.ndarray:
    """Pixels reachable from the image border along True cells."""
    h, w = seed_mask.shape
    out = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def try_push(y: int, x: int) -> None:
        if y < 0 or y >= h or x < 0 or x >= w or out[y, x] or not seed_mask[y, x]:
            return
        out[y, x] = True
        q.append((y, x))

    for x in range(w):
        try_push(0, x)
        try_push(h - 1, x)
    for y in range(h):
        try_push(y, 0)
        try_push(y, w - 1)
    while q:
        y, x = q.popleft()
        try_push(y - 1, x)
        try_push(y + 1, x)
        try_push(y, x - 1)
        try_push(y, x + 1)
    return out


def _bg_from_product(product: np.ndarray, close_px: int) -> np.ndarray:
    """Expand product (seal cracks), then keep only background that touches the frame."""
    bg = ~np.asarray(product, dtype=bool)
    if close_px > 0:
        img = Image.fromarray((bg.astype(np.uint8) * 255), mode="L")
        odd = max(3, int(close_px) | 1)
        img = img.filter(ImageFilter.MinFilter(size=odd))
        bg = np.asarray(img, dtype=np.uint8) >= 128
    return _flood_true(bg)


def _luma_bg_mask(rgb: Image.Image, white_thr: int = 238, close_px: int = 5) -> Image.Image:
    arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    bg = luma >= int(white_thr)
    sat = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    bg |= (luma >= 220) & (sat <= 18)
    bg = _bg_from_product(~bg, close_px)
    return Image.fromarray((bg.astype(np.uint8) * 255), mode="L")


def _rembg_bg_mask(rgb: Image.Image, close_px: int = 5) -> Image.Image | None:
    try:
        from rembg import remove
    except ImportError:
        return None
    cut = remove(rgb.convert("RGBA"))
    if not isinstance(cut, Image.Image):
        cut = Image.fromarray(cut)
    alpha = cut.split()[-1]
    product = np.asarray(alpha, dtype=np.uint8) > 16
    bg = _bg_from_product(product, close_px)
    return Image.fromarray((bg.astype(np.uint8) * 255), mode="L")


def product_inpaint_mask(
    image: Image.Image,
    *,
    dilate_px: int = 8,
    white_thr: int = 238,
    close_px: int = 5,
) -> Image.Image:
    """L mask, white = inpaint (background), black = keep product (solid interior)."""
    rgb = image.convert("RGB")
    mask = _rembg_bg_mask(rgb, close_px=close_px)
    source = "rembg"
    if mask is None:
        mask = _luma_bg_mask(rgb, white_thr=white_thr, close_px=close_px)
        source = "luma-bg"
    if dilate_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=max(3, dilate_px | 1)))
    print(f"   Inpaint mask: {source}  close={close_px}px  dilate={dilate_px}px  (white=fill, interior solid keep)")
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
