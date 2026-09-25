"""Kandinsky 5.0 I2I — product photo as visual condition (not SDXL IP-Adapter)."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .base import ImageBackend


def _to_pil(out) -> Image.Image:
    if isinstance(out, (list, tuple)):
        out = out[0]
    blob = out
    if not isinstance(out, Image.Image):
        blob = (
            getattr(out, "image", None)
            or getattr(out, "images", None)
            or getattr(out, "frames", None)
            or out
        )
    while isinstance(blob, (list, tuple)) and blob:
        blob = blob[0]
    if isinstance(blob, Image.Image):
        return blob.convert("RGB")
    if torch.is_tensor(blob):
        t = blob.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in {1, 3, 4}:
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.dtype != "uint8":
            arr = (arr.clip(0, 1) * 255).astype("uint8") if arr.max() <= 1.0 else arr.clip(0, 255).astype("uint8")
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        return Image.fromarray(arr[..., :3])
    try:
        import numpy as np

        arr = np.array(blob)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != "uint8":
            arr = (arr.clip(0, 1) * 255).astype("uint8") if arr.max() <= 1.0 else arr.clip(0, 255).astype("uint8")
        return Image.fromarray(arr[..., :3] if arr.ndim == 3 else arr).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Kandinsky 5 output type {type(out).__name__}: {exc}") from exc


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


def _compact_k5_prompt(text: str, max_chars: int = 480) -> str:
    """Qwen2.5-VL in K5 I2I has 1024 tokens for template + image + text.

    A long/repetitive LLM prompt truncates *image* tokens → tokens:0 features:198.
    """
    text = " ".join((text or "").split())
    chunks = [p.strip() for p in text.split(",") if p.strip()]
    kept = []
    prev = ""
    seen: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.lower()
        if key == prev:
            continue
        if seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        prev = key
        kept.append(chunk)
    compact = ", ".join(kept)
    if len(compact) > max_chars:
        compact = compact[:max_chars].rsplit(",", 1)[0]
    return compact


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

        prompt = _compact_k5_prompt(prompt)
        if negative_prompt:
            negative_prompt = _compact_k5_prompt(negative_prompt, max_chars=280)
        print(f"   K5 prompt ({len(prompt)} chars): {prompt[:160]}{'…' if len(prompt) > 160 else ''}")

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
        pil = _to_pil(out)
        pil.save(output_path)
        torch.cuda.empty_cache()
        return output_path
