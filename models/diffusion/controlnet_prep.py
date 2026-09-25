"""OpenPose + Depth control maps for SDXL ControlNet.

Preprocessors are loaded, run once, then dropped so SDXL can use RAM.
OpenPose on a flat-lay garment often finds no body — we then zero that scale.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def _resize(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def pose_is_empty(image: Image.Image) -> bool:
    """True only if there is no skeleton (near-black canvas).

    Stick figures have mean luma ~3–5 on 512² — do not use mean brightness.
    """
    hist = image.convert("L").histogram()
    total = sum(hist) or 1
    lit = sum(hist[16:])
    return lit / total < 0.001


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_openpose_map(image: Image.Image) -> Image.Image | None:
    try:
        from controlnet_aux import OpenposeDetector
    except ImportError:
        print("⚠️  controlnet-aux not installed — skip OpenPose. pip install controlnet-aux")
        return None
    print("   ControlNet: running OpenPose…")
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    try:
        mapped = detector(image, include_hand=True, include_face=True)
    except TypeError:
        mapped = detector(image)
    finally:
        del detector
    if mapped is None:
        return None
    if not isinstance(mapped, Image.Image):
        mapped = Image.fromarray(mapped)
    mapped = mapped.convert("RGB")
    if pose_is_empty(mapped):
        print("   ControlNet: OpenPose found no body (typical for flat-lay) — pose scale 0")
        return mapped
    return mapped


def build_depth_map(image: Image.Image) -> Image.Image | None:
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("⚠️  transformers missing — skip Depth")
        return None
    print("   ControlNet: running depth (DPT-Hybrid)…")
    estimator = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    try:
        pred = estimator(image)
    finally:
        del estimator
    depth = pred.get("depth") if isinstance(pred, dict) else pred
    if depth is None:
        return None
    if not isinstance(depth, Image.Image):
        import numpy as np

        arr = np.array(depth)
        if arr.ndim == 2:
            arr = arr.astype("float32")
            arr = (arr - arr.min()) / (max(arr.max() - arr.min(), 1e-6))
            arr = (arr * 255).astype("uint8")
            depth = Image.fromarray(arr, mode="L")
        else:
            depth = Image.fromarray(arr)
    return depth.convert("RGB")


def knockout_studio_bg(image: Image.Image, threshold: int = 238) -> Image.Image:
    """Drop near-white packshot backdrop so Canny does not trace a vertical photo frame."""
    import numpy as np

    arr = np.array(image.convert("RGB"))
    white = (
        (arr[:, :, 0] >= threshold)
        & (arr[:, :, 1] >= threshold)
        & (arr[:, :, 2] >= threshold)
    )
    arr[white] = 0
    return Image.fromarray(arr)


def fit_contain(
    image: Image.Image,
    width: int,
    height: int,
    fill=(0, 0, 0),
    align: str = "center",
    height_frac: float = 0.82,
) -> Image.Image:
    """Place a tall product on a wider canvas without stretching (letterbox)."""
    image = knockout_studio_bg(image.convert("RGB"))
    w, h = image.size
    box_h = max(1, int(height * float(height_frac)))
    scale = min(width / max(w, 1), box_h / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    margin = int(width * 0.10)
    side = (align or "center").lower()
    if side == "left":
        x = margin
    elif side == "right":
        x = max(margin, width - nw - margin)
    else:
        x = (width - nw) // 2
    y = (height - nh) // 2
    canvas.paste(resized, (x, y))
    return canvas


def build_canny_map(
    image: Image.Image,
    low: int = 60,
    high: int = 160,
) -> Image.Image | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("⚠️  opencv not installed — skip Canny. pip install opencv-python-headless")
        return None
    print("   ControlNet: running Canny…")
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, int(low), int(high))
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb)


def build_control_maps(
    source_path: str | Path,
    width: int,
    height: int,
    modes: list[str],
    save_dir: Path | None = None,
    canny_path: str | Path | None = None,
    canny_low: int = 80,
    canny_high: int = 200,
    canny_align: str = "center",
) -> dict[str, Image.Image]:
    src = load_rgb(source_path) if source_path and Path(source_path).is_file() else None
    maps: dict[str, Image.Image] = {}
    wanted = {str(m).lower() for m in modes}

    if src is not None and ("openpose" in wanted or "pose" in wanted):
        pose = build_openpose_map(_resize(src, width, height))
        if pose is not None:
            maps["openpose"] = _resize(pose, width, height)
    if src is not None and "depth" in wanted:
        depth = build_depth_map(_resize(src, width, height))
        if depth is not None:
            maps["depth"] = _resize(depth, width, height)
    if "canny" in wanted:
        cpath = canny_path or source_path
        if cpath and Path(cpath).is_file():
            fitted = fit_contain(load_rgb(cpath), width, height, align=canny_align)
            canny = build_canny_map(fitted, low=canny_low, high=canny_high)
            if canny is not None:
                maps["canny"] = canny

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, img in maps.items():
            out = save_dir / f"{name}.png"
            img.save(out)
            print(f"   ControlNet map → {out}")
    return maps
