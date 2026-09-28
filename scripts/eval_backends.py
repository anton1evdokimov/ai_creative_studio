#!/usr/bin/env python3
"""Compare kandinsky5 / flux2 / sdxl_inpaint on one product photo.

Writes CSV: backend, seconds, peak_vram_gb, clip_t, clip_i, dino_i, cer, path, error
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="commercial product photo, wooden table, soft daylight")
    p.add_argument("--out_dir", default="generated/eval_backends")
    p.add_argument(
        "--backends",
        default="kandinsky5,flux2,sdxl_inpaint",
        help="comma list",
    )
    args = p.parse_args()
    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise SystemExit(f"missing image {image}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from models.diffusion.config import load_diffusion_config, load_ranking_config
    from models.diffusion.factory import unload_image_backend
    from models.ranking.cer import score_cer
    from models.ranking.clip_metrics import CLIPMetrics, unload_clip_metrics

    base = load_diffusion_config()
    names = [x.strip() for x in args.backends.split(",") if x.strip()]
    rows = []

    for name in names:
        unload_image_backend()
        path = out_dir / f"{name}.png"
        err = ""
        t0 = time.time()
        vram = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            cfg = dict(base)
            cfg["backend"] = name
            cfg["model"] = "sdxl" if "sdxl" in name else name
            if name == "kandinsky5":
                from models.diffusion.kandinsky_backend import Kandinsky5Backend as Cls
            elif name == "flux2":
                from models.diffusion.flux2_backend import Flux2Backend as Cls
            elif name in {"sdxl_inpaint", "inpaint"}:
                from models.diffusion.sdxl_inpaint_backend import SDXLInpaintBackend as Cls
            else:
                raise ValueError(f"unknown backend {name}")
            pipe = Cls(cfg)
            pipe.generate(
                args.prompt,
                str(path),
                seed=0,
                control_image=str(image),
                ip_adapter_image=str(image),
            )
            del pipe
            unload_image_backend()
            vram = _vram_gb()
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"⚠️  {name}: {err}")
        dt = round(time.time() - t0, 2)
        clip_i = cer = clip_t = dino_i = ""
        if path.is_file() and not err:
            try:
                rcfg = load_ranking_config()
                clipper = CLIPMetrics(rcfg.get("clip_model") or "openai/clip-vit-large-patch14", use_aesthetic=False)
                m = clipper.score(path, args.prompt, product_image_path=str(image))
                clip_i = m.clip_i
                clip_t = m.clip_t
                unload_clip_metrics()
            except Exception as exc:
                err = (err + " | " if err else "") + f"CLIP {exc}"
                clip_t = ""
            dino_i = ""
            try:
                from models.ranking.dino_metrics import DinoMetrics, unload_dino_metrics

                dino = DinoMetrics()
                dino_i, _ = dino.similarity(str(path), str(image))
                unload_dino_metrics()
            except Exception:
                pass
            try:
                c, _, _ = score_cer(path, {"product_name": image.stem}, "rus+eng")
                cer = c
            except Exception:
                pass
        row = {
            "backend": name,
            "seconds": dt,
            "peak_vram_gb": vram if vram is not None else "",
            "clip_i": clip_i if clip_i != "" else "",
            "clip_t": clip_t if clip_t != "" else "",
            "dino_i": dino_i if dino_i != "" else "",
            "cer": cer if cer != "" else "",
            "path": str(path) if path.is_file() else "",
            "error": err,
        }
        rows.append(row)
        print(row)

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
