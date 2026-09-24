#!/usr/bin/env python3
"""Ablation demo: SDXL ControlNet OpenPose + Depth.

Puts pose_ref, OpenPose map, Depth map, and 4 generations (none / pose / depth / both)
on one contact sheet. This is the figure you want in a portfolio / interview.

Images (required):
  data/input/pose_ref.jpg   full-body person, 3/4 view, visible shoulders
  data/input/hoodie.webp    product photo (caption only; not the pose)

    python scripts/demo_controlnet.py \\
      --pose data/input/pose_ref.jpg \\
      --prompt "pink oversized hoodie, fashion lookbook, studio, photorealistic"

Needs: controlnet-aux, opencv-python-headless, SDXL base + CN weights, GPU RAM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _label(img: Image.Image, text: str) -> Image.Image:
    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return canvas


def _sheet(cells: list[tuple[str, Image.Image]], cols: int, cell: int) -> Image.Image:
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
    for i, (title, im) in enumerate(cells):
        thumb = im.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        thumb = _label(thumb, title)
        r, c = divmod(i, cols)
        grid.paste(thumb, (c * cell, r * cell))
    return grid


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pose", default="data/input/pose_ref.jpg")
    p.add_argument("--prompt", default="pink oversized hoodie on a model, fashion lookbook, studio lighting, photorealistic")
    p.add_argument("--out_dir", default="generated/controlnet_demo")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pose = Path(args.pose)
    if not pose.is_file():
        raise SystemExit(
            f"Missing {pose}. Put a FULL-BODY 3/4 person photo there "
            "(not a flat-lay hoodie). OpenPose needs a skeleton."
        )

    from models.diffusion.config import load_diffusion_config
    from models.diffusion.controlnet_prep import build_control_maps, pose_is_empty
    from models.diffusion.sdxl_backend import SDXLBackend

    cfg = load_diffusion_config()
    cfg["controlnet"] = {
        **(cfg.get("controlnet") or {}),
        "enabled": True,
        "modes": ["openpose", "depth"],
        "save_maps": True,
        "image": str(pose.resolve()),
        "openpose_scale": 0.55,
        "depth_scale": 0.65,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out.resolve())
    w, h = int(cfg["width"]), int(cfg["height"])

    maps = build_control_maps(pose, w, h, ["openpose", "depth"], save_dir=out / "maps")
    pose_img = Image.open(pose).convert("RGB").resize((w, h))
    openpose = maps.get("openpose")
    depth = maps.get("depth")
    if openpose is None or pose_is_empty(openpose):
        print("⚠️  OpenPose map is empty — pose_ref has no body. Demo will look like Depth-only.")
    if depth:
        depth.save(out / "maps" / "depth.png")

    backend = SDXLBackend(cfg)
    variants = [
        ("00_none", 0.0, 0.0),
        ("01_openpose", 0.85, 0.0),
        ("02_depth", 0.0, 0.85),
        ("03_both", 0.55, 0.65),
    ]
    gens = []
    for name, ps, ds in variants:
        path = str(out / f"{name}.png")
        backend.generate(
            args.prompt,
            path,
            seed=args.seed,
            control_image=str(pose.resolve()),
            openpose_scale=ps,
            depth_scale=ds,
            negative_prompt="cartoon, extra limbs, deformed, blurry, watermark",
        )
        gens.append((name, Image.open(path)))

    cells = [
        ("pose_ref", pose_img),
        ("openpose", openpose or Image.new("RGB", (w, h))),
        ("depth", depth or Image.new("RGB", (w, h))),
        *gens,
    ]
    sheet = _sheet(cells, cols=4, cell=min(w, 384))
    sheet_path = out / "ablation_sheet.png"
    sheet.save(sheet_path)
    print(f"\nSheet → {sheet_path}")
    print("Read it left-to-right: maps, then none / OpenPose / Depth / both (same seed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
