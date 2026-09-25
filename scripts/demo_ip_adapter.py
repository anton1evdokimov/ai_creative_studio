#!/usr/bin/env python3
"""Same product reference + several scene prompts → IP-Adapter Plus + SDXL.

Does not touch ControlNet. Isolated identity check.

    python scripts/demo_ip_adapter.py --image data/input/hoodie.webp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROMPTS = [
    "pink oversized hoodie on a model, minimal white studio, softbox lighting, fashion lookbook, photorealistic",
    "pink oversized hoodie, golden hour street, cinematic, shallow depth of field, photorealistic",
    "pink oversized hoodie on a hanger, boutique interior, warm tungsten, product catalog, photorealistic",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="data/input/hoodie.webp")
    p.add_argument("--out_dir", default="generated/ip_adapter_demo")
    p.add_argument("--scale", type=float, default=0.6, help="ip_adapter_scale")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ref = Path(args.image)
    if not ref.is_file():
        raise SystemExit(f"Missing reference: {ref}")

    from models.diffusion.config import load_diffusion_config
    from models.diffusion.sdxl_backend import SDXLBackend

    cfg = load_diffusion_config()
    cfg["model"] = "sdxl"
    cfg["num_inference_steps"] = max(int(cfg.get("num_inference_steps") or 20), 20)
    cfg["guidance_scale"] = max(float(cfg.get("guidance_scale") or 5.0), 5.0)
    cfg["controlnet"] = {**(cfg.get("controlnet") or {}), "enabled": False}
    cfg["ip_adapter"] = {
        **(cfg.get("ip_adapter") or {}),
        "enabled": True,
        "scale": float(args.scale),
        "image": str(ref.resolve()),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out.resolve())

    backend = SDXLBackend(cfg)
    for i, prompt in enumerate(PROMPTS):
        path = str(out / f"{i:02d}_scene.png")
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:80]}")
        backend.generate(
            prompt,
            path,
            seed=args.seed,
            ip_adapter_image=str(ref.resolve()),
            ip_adapter_scale=float(args.scale),
            negative_prompt="cartoon, extra limbs, deformed, blurry, watermark",
        )
    print(f"\nSame ref + different prompts → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
