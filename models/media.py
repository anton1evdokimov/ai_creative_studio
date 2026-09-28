"""Upload sniffing + save format / canvas aspect from the product photo."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 40_000_000

_FMT_EXT = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".png",
    "GIF": ".png",
}

_OUT_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def output_ext_for(product_image: str) -> str:
    ext = Path(product_image or "").suffix.lower()
    if ext == ".jpeg":
        return ".jpg"
    if ext in _OUT_EXT:
        return ext
    return ".png"


def save_rgb(image: Image.Image, path: str | Path) -> None:
    p = Path(path)
    rgb = image.convert("RGB")
    ext = p.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        rgb.save(p, "JPEG", quality=92, optimize=True)
    elif ext == ".webp":
        rgb.save(p, "WEBP", quality=90, method=4)
    else:
        rgb.save(p, "PNG", optimize=True)


def ingest_image_bytes(data: bytes, filename: str, dest: Path) -> Path:
    try:
        im = Image.open(BytesIO(data))
        im.load()
    except Exception as exc:
        raise ValueError(f"Not a readable image: {exc}") from exc
    ext = Path(filename or "").suffix.lower()
    if ext not in _OUT_EXT | {".bmp", ".tif", ".tiff", ".gif"}:
        ext = _FMT_EXT.get((im.format or "PNG").upper(), ".png")
    if ext in {".tif", ".tiff", ".gif", ".bmp"}:
        ext = ".png"
    dest = dest.with_suffix(ext)
    save_rgb(im.convert("RGB"), dest)
    return dest
