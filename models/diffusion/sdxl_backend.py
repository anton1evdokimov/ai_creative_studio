from pathlib import Path
import platform

import torch

from .base import ImageBackend
from .config import resolve_hf_model


_TEMPLATE_MARKERS = (
    "you must include",
    "a single long string",
    "comma separated, english only",
    "fine-tuned for flux",
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            try:
                built = bool(mps.is_built())
            except Exception:
                built = True
            try:
                available = bool(mps.is_available())
            except Exception:
                available = built
            if available or built:
                return "mps"
        print("⚠️  MPS not detected — SDXL would run on CPU (very slow)")
    return "cpu"


def compact_sdxl_prompt(prompt: str, max_chars: int = 320) -> str:
    """SDXL CLIP is 77 tokens. Keep a short prompt; drop leaked LLM instructions."""
    text = " ".join((prompt or "").split())
    low = text.lower()
    if any(marker in low for marker in _TEMPLATE_MARKERS):
        for sep in ("You MUST include:", "MUST include:", "include:"):
            idx = low.find(sep.lower())
            if idx != -1:
                text = text[idx + len(sep) :].strip(" ,")
                break
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(",", 1)[0]
    return text


class SDXLBackend(ImageBackend):
    """SDXL / SDXL-Turbo via Diffusers. Much smaller on disk than FLUX.1 (~7GB vs ~20GB+)."""

    def __init__(self, config: dict):
        self.config = config
        self.device = _pick_device()
        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        from diffusers import AutoPipelineForText2Image

        model_id = resolve_hf_model(self.config["model"])
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        print(f"Loading SDXL ({model_id}) via Diffusers on {self.device}")

        kwargs = {"torch_dtype": dtype}
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, variant="fp16", **kwargs
            )
        except Exception:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)

        pipe = self._maybe_load_lora(pipe)

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            try:
                pipe.to(self.device)
            except Exception as exc:
                print(f"⚠️  Could not move SDXL to {self.device} ({exc}); using CPU")
                self.device = "cpu"
                pipe.to("cpu")

        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        return pipe

    def _maybe_load_lora(self, pipe):
        lora = self.config.get("lora") or {}
        enabled = bool(lora.get("enabled"))
        raw_path = str(lora.get("path") or "").strip()
        if not enabled or not raw_path:
            return pipe

        path = Path(raw_path)
        if not path.exists():
            print(f"⚠️  LoRA enabled but path not found: {path} — generating without LoRA")
            return pipe

        name = str(self.config.get("model", "")).lower()
        if "turbo" in name:
            print("⚠️  LoRA was likely trained on SDXL base; Turbo may ignore or distort it. Prefer diffusion.model: sdxl")

        try:
            if path.is_file():
                pipe.load_lora_weights(str(path.parent), weight_name=path.name)
            else:
                pipe.load_lora_weights(str(path))
            scale = float(lora.get("scale") or 1.0)
            if hasattr(pipe, "set_adapters"):
                try:
                    pipe.set_adapters(["default"], adapter_weights=[scale])
                except Exception:
                    if hasattr(pipe, "fuse_lora"):
                        pipe.fuse_lora(lora_scale=scale)
            elif hasattr(pipe, "fuse_lora"):
                pipe.fuse_lora(lora_scale=scale)
            print(f"🧩 Loaded LoRA from {path}  (scale={scale})")
        except Exception as exc:
            print(f"⚠️  Failed to load LoRA ({type(exc).__name__}: {exc}) — generating without LoRA")
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

        prompt = compact_sdxl_prompt(prompt)
        trigger = str((self.config.get("lora") or {}).get("trigger") or "").strip()
        if (self.config.get("lora") or {}).get("enabled") and trigger:
            if trigger.lower() not in prompt.lower():
                prompt = compact_sdxl_prompt(f"{trigger}, {prompt}")
        negative = compact_sdxl_prompt(negative_prompt or "", max_chars=280)
        print(f"   SDXL prompt ({len(prompt)} chars): {prompt[:160]}{'…' if len(prompt) > 160 else ''}")

        generator = None
        if seed is not None:
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(gen_device).manual_seed(seed)

        pipe_kwargs = dict(
            prompt=prompt,
            height=int(self.config["height"]),
            width=int(self.config["width"]),
            guidance_scale=float(self.config["guidance_scale"]),
            num_inference_steps=int(self.config["num_inference_steps"]),
            generator=generator,
        )
        if negative:
            pipe_kwargs["negative_prompt"] = negative

        image = self.pipe(**pipe_kwargs).images[0]
        image.save(output_path)

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return output_path
