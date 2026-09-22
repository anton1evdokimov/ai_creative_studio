"""Lightweight image-quality heuristics (no ML needed).

Complements the VLM scorer with cheap, deterministic checks:
  - file exists and opens correctly
  - meets target width/height (no silent downscaling bugs)
  - not under-sized / saved with catastrophic JPEG compression
  - basic Laplacian-variance blur detection (PIL-based)

Returns a 0..1 quality factor that can be multiplied with the VLM overall score.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def _laplacian_variance(gray_img) -> float:
    """Very small 3x3 convolution blur estimate (no numpy required for path)."""
    w, h = gray_img.size
    if w < 32 or h < 32:
        return 0.0
    px = gray_img.load()
    total = 0.0
    samples = 0
    # Sample a grid to keep it cheap for large images
    step_x = max(1, w // 160)
    step_y = max(1, h // 160)
    for y in range(1, h - 1, step_y):
        for x in range(1, w - 1, step_x):
            # 3x3 Laplacian kernel: [0,1,0;1,-4,1;0,1,0]
            v = (
                px[x, y - 1] + px[x - 1, y] + px[x + 1, y] + px[x, y + 1] - 4 * px[x, y]
            )
            total += v * v
            samples += 1
    return total / max(1, samples)


def analyze_quality(
    image_path: str | Path,
    expected_width: int = 832,
    expected_height: int = 832,
    expected_mode: str = "RGB",
) -> dict:
    p = Path(image_path)
    result = {
        "ok": False,
        "exists": p.exists(),
        "opens": False,
        "width": 0,
        "height": 0,
        "mode": "",
        "size_bytes": 0,
        "meets_resolution": False,
        "blur_score": 0.0,
        "compression_ratio": 0.0,
        "quality_factor": 0.0,  # 0..1
        "notes": [],
    }
    if not result["exists"]:
        result["notes"].append("file missing")
        return result
    try:
        result["size_bytes"] = p.stat().st_size
        with Image.open(p) as im:
            result["opens"] = True
            result["width"], result["height"] = im.size
            result["mode"] = im.mode
            result["meets_resolution"] = (
                result["width"] >= expected_width and result["height"] >= expected_height
            )
            if not result["meets_resolution"]:
                result["notes"].append(
                    f"low res: {result['width']}x{result['height']} vs target {expected_width}x{expected_height}"
                )
            gray = im.convert("L")
            lv = _laplacian_variance(gray)
            # Normalise: ~100 = very blurry, ~5000+ = sharp product shot
            # Rescale to 0..1 with sigmoid-ish ramp
            import math
            result["blur_score"] = round(1.0 - 1.0 / (1.0 + math.exp((lv - 800.0) / 300.0)), 3)
            # Expected uncompressed bytes (RGB): W*H*3
            uncompressed_bytes = result["width"] * result["height"] * (3 if expected_mode == "RGB" else 4)
            if result["size_bytes"] and uncompressed_bytes:
                result["compression_ratio"] = round(uncompressed_bytes / result["size_bytes"], 2)
            # PNG for 832x832 product shots typically compresses 3x-6x cleanly; >25x suggests content loss.
            if result["compression_ratio"] > 25:
                result["notes"].append("very high compression ratio, likely low-detail or OOD artifacts")
    except Exception as exc:
        result["notes"].append(f"open/parse error: {type(exc).__name__}: {exc}")

    # Combine sub-factors
    factor = 0.0
    if result["opens"]:
        factor += 0.4
        factor += 0.25 if result["meets_resolution"] else 0.0
        factor += 0.25 * result["blur_score"]
        # Compression sanity bonus: ratio in 2..20 is healthy
        factor += 0.1 * (1.0 if 2.0 <= result["compression_ratio"] <= 25.0 else max(0.0, 1.0 - abs(result["compression_ratio"] - 10.0) / 15.0))
    result["quality_factor"] = round(max(0.0, min(1.0, factor)), 3)
    result["ok"] = result["opens"] and result["quality_factor"] >= 0.45
    return result
