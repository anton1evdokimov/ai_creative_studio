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
    if len(text) <= max_chars:
        return text
    # Keep the subject (text before Scene/Lighting) when truncating
    cut_at = None
    for marker in (". Scene:", ". Lighting:", " Scene:"):
        idx = text.find(marker)
        if idx != -1:
            cut_at = idx
            break
    if cut_at and cut_at < max_chars:
        head = text[: cut_at + 1].strip()
        rest = text[cut_at + 1 :].strip()
        budget = max_chars - len(head) - 1
        if budget > 20:
            rest = rest[:budget].rsplit(",", 1)[0]
            return f"{head} {rest}".strip()
        return head[:max_chars]
    return text[:max_chars].rsplit(",", 1)[0]


class SDXLBackend(ImageBackend):
    """SDXL / SDXL-Turbo via Diffusers. Optional OpenPose + Depth ControlNet."""

    def __init__(self, config: dict):
        self.config = config
        self.device = _pick_device()
        self._cn_order: list[str] = []
        self._maps_cache: dict[str, dict] = {}
        self._ip_ready = False
        self.pipe = self._load_pipeline()

    def _cn_cfg(self) -> dict:
        return self.config.get("controlnet") or {}

    def _cn_enabled(self) -> bool:
        return bool(self._cn_cfg().get("enabled"))

    def _load_pipeline(self):
        from diffusers import AutoPipelineForText2Image

        if self._cn_enabled():
            pipe = self._load_controlnet_pipeline()
            if pipe is not None:
                return pipe
            print("⚠️  ControlNet failed to load — falling back to plain SDXL")

        from .config import HF_MODELS

        if self._ip_enabled():
            model_id = HF_MODELS["sdxl"]
            print("⚠️  IP-Adapter Plus uses SDXL base (Turbo UNet is incompatible)")
        else:
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
        pipe = self._maybe_load_ip_adapter(pipe)
        return self._place_pipe(pipe)

    def _load_controlnet_pipeline(self):
        try:
            return self._load_controlnet_pipeline_inner()
        except Exception as exc:
            print(f"⚠️  ControlNet load failed ({type(exc).__name__}: {exc})")
            self._cn_order = []
            return None

    def _load_controlnet_pipeline_inner(self):
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        from .config import HF_MODELS

        cfg = self._cn_cfg()
        modes = [str(m).lower() for m in (cfg.get("modes") or ["openpose", "depth"])]
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        nets = []
        order = []

        if any(m in {"openpose", "pose"} for m in modes):
            mid = cfg.get("openpose_model") or "thibaud/controlnet-openpose-sdxl-1.0"
            print(f"Loading ControlNet OpenPose ({mid})")
            nets.append(ControlNetModel.from_pretrained(mid, torch_dtype=dtype))
            order.append("openpose")
        if "depth" in modes:
            mid = cfg.get("depth_model") or "diffusers/controlnet-depth-sdxl-1.0"
            print(f"Loading ControlNet Depth ({mid})")
            nets.append(ControlNetModel.from_pretrained(mid, torch_dtype=dtype))
            order.append("depth")
        if not nets:
            return None

        controlnet = nets[0] if len(nets) == 1 else nets
        base_id = HF_MODELS["sdxl"]
        name = str(self.config.get("model", "")).lower()
        if "turbo" in name:
            print("⚠️  ControlNet uses SDXL base (Turbo is a poor ControlNet host). steps≥20, guidance≥5")
        print(f"Loading SDXL+ControlNet ({base_id}) on {self.device}")
        kwargs = {"torch_dtype": dtype, "controlnet": controlnet}
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_id, variant="fp16", **kwargs
            )
        except Exception:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base_id, **kwargs)

        self._cn_order = order
        pipe = self._maybe_load_lora(pipe)
        pipe = self._maybe_load_ip_adapter(pipe)
        return self._place_pipe(pipe)

    def _place_pipe(self, pipe):
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            try:
                pipe.to(self.device)
            except Exception as exc:
                print(f"⚠️  Could not move SDXL to {self.device} ({exc}); using CPU")
                self.device = "cpu"
                pipe.to("cpu")

        if hasattr(pipe, "enable_attention_slicing") and not getattr(self, "_ip_ready", False):
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

    def _ip_cfg(self) -> dict:
        return self.config.get("ip_adapter") or {}

    def _ip_enabled(self) -> bool:
        return bool(self._ip_cfg().get("enabled"))

    def _maybe_load_ip_adapter(self, pipe):
        self._ip_ready = False
        if not self._ip_enabled():
            return pipe
        name = str(self.config.get("model", "")).lower()
        if "turbo" in name and not self._cn_enabled():
            print("⚠️  IP-Adapter Plus is trained for SDXL base; Turbo-only UNet may fail. Prefer diffusion.model: sdxl")
        try:
            from .adapters import attach_ip_adapter_plus

            scale = float(self._ip_cfg().get("scale") or 0.6)
            pipe = attach_ip_adapter_plus(pipe, scale=scale)
            self._ip_ready = True
            print(f"🧩 Loaded IP-Adapter Plus (scale={scale})")
        except Exception as exc:
            print(f"⚠️  IP-Adapter Plus failed ({type(exc).__name__}: {exc}) — generating without it")
        return pipe

    def _ip_source(self, extra: dict) -> str:
        cfg_img = str(self._ip_cfg().get("image") or "").strip()
        if cfg_img and Path(cfg_img).is_file():
            return cfg_img
        extra_img = str(extra.get("ip_adapter_image") or "").strip()
        if extra_img and Path(extra_img).is_file():
            return extra_img
        return cfg_img or extra_img

    def _control_source(self, extra: dict) -> str:
        cfg_img = str(self._cn_cfg().get("image") or "").strip()
        if cfg_img and Path(cfg_img).is_file():
            return cfg_img
        extra_img = str(extra.get("control_image") or "").strip()
        if extra_img and Path(extra_img).is_file():
            return extra_img
        return cfg_img or extra_img

    def _maps_for(self, source: str, width: int, height: int) -> dict:
        key = f"{source}:{width}x{height}"
        if key in self._maps_cache:
            return self._maps_cache[key]
        from .controlnet_prep import build_control_maps, pose_is_empty

        cfg = self._cn_cfg()
        save_dir = None
        if cfg.get("save_maps"):
            save_dir = Path(self.config["output_dir"]) / "control"
        maps = build_control_maps(
            source,
            width,
            height,
            cfg.get("modes") or ["openpose", "depth"],
            save_dir=save_dir,
        )
        maps["_pose_empty"] = bool(maps.get("openpose") and pose_is_empty(maps["openpose"]))
        self._maps_cache[key] = maps
        return maps

    def _cn_images_and_scales(self, maps: dict, extra: dict | None = None) -> tuple:
        extra = extra or {}
        cfg = self._cn_cfg()
        images = []
        scales = []
        w = int(self.config["width"])
        h = int(self.config["height"])
        blank = None
        for name in self._cn_order:
            img = maps.get(name)
            if img is None:
                from PIL import Image as PILImage

                blank = blank or PILImage.new("RGB", (w, h), (0, 0, 0))
                images.append(blank)
                scales.append(0.0)
                continue
            images.append(img)
            if name == "openpose":
                if extra.get("openpose_scale") is not None:
                    scale = float(extra["openpose_scale"])
                else:
                    scale = float(cfg.get("openpose_scale") or 0.55)
                    if maps.get("_pose_empty"):
                        scale = 0.0
            else:
                scale = (
                    float(extra["depth_scale"])
                    if extra.get("depth_scale") is not None
                    else float(cfg.get("depth_scale") or 0.65)
                )
            scales.append(scale)
        if not images:
            return None, None
        if len(images) == 1:
            return images[0], scales[0]
        return images, scales

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

        steps = int(self.config["num_inference_steps"])
        guidance = float(self.config["guidance_scale"])
        use_cn = self._cn_enabled() and self._cn_order
        use_ip = self._ip_enabled() and getattr(self, "_ip_ready", False)
        if use_cn or use_ip:
            name = str(self.config.get("model", "")).lower()
            if "turbo" in name:
                steps = max(steps, 20)
                if guidance < 1.0:
                    guidance = 5.0

        pipe_kwargs = dict(
            prompt=prompt,
            height=int(self.config["height"]),
            width=int(self.config["width"]),
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        if negative:
            pipe_kwargs["negative_prompt"] = negative

        if use_cn:
            source = self._control_source(extra)
            images, scales = None, None
            if source and Path(source).is_file():
                maps = self._maps_for(source, int(self.config["width"]), int(self.config["height"]))
                images, scales = self._cn_images_and_scales(maps, extra)
            if images is None:
                from PIL import Image as PILImage

                w, h = int(self.config["width"]), int(self.config["height"])
                blanks = [PILImage.new("RGB", (w, h), (0, 0, 0)) for _ in self._cn_order]
                scales = [0.0] * len(self._cn_order)
                images = blanks[0] if len(blanks) == 1 else blanks
                print("⚠️  ControlNet: no usable maps — scales=0")
            pipe_kwargs["image"] = images
            pipe_kwargs["controlnet_conditioning_scale"] = scales
            print(f"   ControlNet scales: {self._cn_order} → {scales}")

        if use_ip:
            ref = self._ip_source(extra)
            scale = extra.get("ip_adapter_scale")
            if scale is None:
                scale = self._ip_cfg().get("scale") or 0.6
            scale = float(scale)
            self.pipe.set_ip_adapter_scale(scale)
            if ref and Path(ref).is_file():
                from .adapters import load_rgb

                pipe_kwargs["ip_adapter_image"] = load_rgb(ref)
                print(f"   IP-Adapter Plus scale={scale}  ref={Path(ref).name}")
            else:
                self.pipe.set_ip_adapter_scale(0.0)
                print("⚠️  IP-Adapter Plus enabled but no reference image — scale=0")

        image = self.pipe(**pipe_kwargs).images[0]
        image.save(output_path)

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return output_path
