"""Kandinsky 5.0 I2I — product photo as visual condition (not SDXL IP-Adapter)."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .base import ImageBackend


K5_I2I_DEFAULT = "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
K5_SIZES = (
    (1024, 1024),
    (640, 1408),
    (1408, 640),
    (768, 1280),
    (1280, 768),
    (896, 1152),
    (1152, 896),
)


def _snap_hw(width: int, height: int) -> tuple[int, int]:
    target = width / max(height, 1)
    return min(K5_SIZES, key=lambda hw: abs(hw[0] / hw[1] - target))


def _k5_cfg(config: dict) -> dict:
    return config.get("kandinsky") if isinstance(config.get("kandinsky"), dict) else {}


class Kandinsky5Backend(ImageBackend):
    def __init__(self, config: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("Kandinsky 5 I2I needs CUDA.")
        self.config = config
        self.pipe = self._load()

    def _load(self):
        try:
            from diffusers import Kandinsky5I2IPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers has no Kandinsky5I2IPipeline — upgrade: pip install -U diffusers"
            ) from exc

        k5 = _k5_cfg(self.config)
        model_id = str(k5.get("model") or K5_I2I_DEFAULT)
        print(f"Loading Kandinsky 5 I2I ({model_id}) on cuda")
        pipe = Kandinsky5I2IPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        return pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
        **extra,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        k5 = _k5_cfg(self.config)
        ref = (
            str(extra.get("ip_adapter_image") or "").strip()
            or str(extra.get("control_image") or "").strip()
            or str(k5.get("image") or "").strip()
        )
        if not ref or not Path(ref).is_file():
            raise FileNotFoundError("Kandinsky 5 I2I needs a product photo (same --image as the pipeline).")

        image = Image.open(ref).convert("RGB")
        w = int(k5.get("width") or self.config.get("width") or 1280)
        h = int(k5.get("height") or self.config.get("height") or 768)
        w, h = _snap_hw(w, h)

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        kwargs = dict(
            image=image,
            prompt=prompt,
            height=h,
            width=w,
            num_inference_steps=int(k5.get("num_inference_steps") or 50),
            guidance_scale=float(k5.get("guidance_scale") or 3.5),
            generator=generator,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        print(f"   Kandinsky5 I2I  {Path(ref).name}  {w}x{h}  steps={kwargs['num_inference_steps']}")

        out = self.pipe(**kwargs)
        pil = None
        if getattr(out, "images", None):
            pil = out.images[0]
        elif getattr(out, "frames", None):
            frames = out.frames[0]
            pil = frames[0] if isinstance(frames, (list, tuple)) else frames
        if pil is None:
            raise RuntimeError("Kandinsky 5 returned no image")
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        pil.save(output_path)
        torch.cuda.empty_cache()
        return output_path
