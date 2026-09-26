"""FLUX.2 Klein — product photo as reference tokens (not SDXL IP-Adapter, not img2img)."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .base import ImageBackend
from .kandinsky_backend import _TEXT_NEG, _compact_k5_prompt, _label_block, _letterbox


FLUX2_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"


def _f2_cfg(config: dict) -> dict:
    return config.get("flux2") if isinstance(config.get("flux2"), dict) else {}


class Flux2Backend(ImageBackend):
    def __init__(self, config: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 needs CUDA.")
        self.config = config
        self.pipe = self._load()

    def _load(self):
        f2 = _f2_cfg(self.config)
        model_id = str(f2.get("model") or FLUX2_DEFAULT)
        print(f"Loading FLUX.2 ({model_id}) on cuda")
        try:
            from diffusers import Flux2KleinPipeline as PipeCls
        except ImportError as exc:
            raise RuntimeError(
                "diffusers has no Flux2KleinPipeline — pip install -U git+https://github.com/huggingface/diffusers.git"
            ) from exc
        if "dev" in model_id.lower() and "klein" not in model_id.lower():
            try:
                from diffusers import Flux2Pipeline as PipeCls
            except ImportError as exc:
                raise RuntimeError("Need Flux2Pipeline for FLUX.2-dev") from exc
        pipe = PipeCls.from_pretrained(model_id, torch_dtype=torch.bfloat16)
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
        f2 = _f2_cfg(self.config)
        ref = (
            str(extra.get("ip_adapter_image") or "").strip()
            or str(extra.get("control_image") or "").strip()
            or str(f2.get("image") or "").strip()
        )
        if not ref or not Path(ref).is_file():
            raise FileNotFoundError("FLUX.2 needs a product photo (same --image as the pipeline).")

        w = int(f2.get("width") or 768)
        h = int(f2.get("height") or 1280)
        image = _letterbox(Image.open(ref).convert("RGB"), w, h)

        prompt = _compact_k5_prompt(prompt, max_chars=400)
        lock = _label_block(str(extra.get("label_text") or ""))
        if lock:
            prompt = f"{lock}{prompt}"
        if negative_prompt:
            negative_prompt = _compact_k5_prompt(negative_prompt, max_chars=240)
            if _TEXT_NEG not in negative_prompt:
                negative_prompt = f"{negative_prompt}, {_TEXT_NEG}"
        else:
            negative_prompt = _TEXT_NEG

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        kwargs = dict(
            prompt=prompt,
            image=image,
            height=h,
            width=w,
            num_inference_steps=int(f2.get("num_inference_steps") or 4),
            guidance_scale=float(f2.get("guidance_scale") or 1.0),
            generator=generator,
        )
        print(f"   FLUX.2  {Path(ref).name}  {w}x{h}  steps={kwargs['num_inference_steps']}")
        try:
            out = self.pipe(**kwargs)
        except TypeError:
            kwargs.pop("image", None)
            print("⚠️  This FLUX.2 build ignores image= — T2I only")
            out = self.pipe(**kwargs)
        pil = out.images[0]
        pil.save(output_path)
        torch.cuda.empty_cache()
        return output_path
