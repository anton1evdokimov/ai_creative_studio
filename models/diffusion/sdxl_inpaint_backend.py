"""SDXL inpaint: keep product pixels, generate background from the prompt."""
from __future__ import annotations

from pathlib import Path

import torch

from models.media import save_rgb
from .base import ImageBackend
from .inpaint_mask import letterbox_rgb_mask, product_inpaint_mask, save_mask
from .kandinsky_backend import _snap_hw
from .sdxl_backend import compact_sdxl_prompt, _pick_device


INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class SDXLInpaintBackend(ImageBackend):
    def __init__(self, config: dict):
        self.config = config
        self.device = _pick_device()
        self.pipe = self._load()

    def _cfg(self) -> dict:
        return self.config.get("inpaint") if isinstance(self.config.get("inpaint"), dict) else {}

    def _load(self):
        from diffusers import AutoPipelineForInpainting

        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        model_id = str(self._cfg().get("model") or INPAINT_MODEL)
        print(f"Loading SDXL inpaint ({model_id}) on {self.device}")
        kwargs = {"torch_dtype": dtype}
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(model_id, variant="fp16", **kwargs)
        except Exception:
            pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)
        if self.device == "cuda":
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
        else:
            pipe.to(self.device)
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
        cfg = self._cfg()
        ref = (
            str(extra.get("ip_adapter_image") or extra.get("control_image") or "").strip()
            or str(cfg.get("image") or "").strip()
        )
        if not ref or not Path(ref).is_file():
            raise FileNotFoundError("SDXL inpaint needs a product photo (--image).")

        from PIL import Image

        raw = Image.open(ref).convert("RGB")
        w, h = _snap_hw(*raw.size)
        long = int(cfg.get("long_side") or 1024)
        scale = long / max(w, h)
        w, h = int(round(w * scale / 8) * 8), int(round(h * scale / 8) * 8)
        w, h = max(w, 512), max(h, 512)

        mask = product_inpaint_mask(
            raw,
            dilate_px=int(cfg.get("mask_dilate") or 8),
            white_thr=int(cfg.get("white_thr") or 238),
        )
        image, mask_l = letterbox_rgb_mask(raw, mask, w, h)
        if cfg.get("save_mask", True):
            save_mask(mask_l, Path(output_path).with_name(Path(output_path).stem + "_mask.png"))

        prompt = compact_sdxl_prompt(prompt, max_chars=420)
        negative = compact_sdxl_prompt(
            negative_prompt
            or "cropped product, extra bottles, melted label, text artifacts, watermark",
            max_chars=320,
        )
        gen = None
        if seed is not None:
            gen_device = "cpu" if self.device == "mps" else self.device
            gen = torch.Generator(gen_device).manual_seed(int(seed))

        steps = int(self.config.get("num_inference_steps") or 20)
        guidance = float(self.config.get("guidance_scale") or 5.5)
        strength = float(cfg.get("strength") or 0.92)
        print(f"   SDXL inpaint  {Path(ref).name}  {w}x{h}  strength={strength}  steps={steps}")
        print(f"   prompt ({len(prompt)} chars): {prompt[:160]}")

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            image=image,
            mask_image=mask_l,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            generator=gen,
        ).images[0]
        save_rgb(out, output_path)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return output_path
