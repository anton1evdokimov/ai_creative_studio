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


def build_control_maps(
    source_path: str | Path,
    width: int,
    height: int,
    modes: list[str],
    save_dir: Path | None = None,
) -> dict[str, Image.Image]:
    src = load_rgb(source_path)
    src = _resize(src, width, height)
    maps: dict[str, Image.Image] = {}
    wanted = {str(m).lower() for m in modes}

    if "openpose" in wanted or "pose" in wanted:
        pose = build_openpose_map(src)
        if pose is not None:
            maps["openpose"] = _resize(pose, width, height)
    if "depth" in wanted:
        depth = build_depth_map(src)
        if depth is not None:
            maps["depth"] = _resize(depth, width, height)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, img in maps.items():
            out = save_dir / f"{name}.png"
            img.save(out)
            print(f"   ControlNet map → {out}")
    return maps
