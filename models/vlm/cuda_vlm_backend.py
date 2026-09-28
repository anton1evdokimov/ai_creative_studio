"""CUDA VLM via transformers (Qwen2-VL). Same chat_with_image API as MLXVLMBackend."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


class CUDAVLMBackend:
    def __init__(self, model_name: str, max_tokens: int = 900, temperature: float = 0.3):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._boot_time = None
        self._parked = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        t0 = time.time()
        print(f"   👁️  Loading CUDA VLM: {self.model_name} (first run may download)")
        name = self.model_name.lower()
        if "qwen2.5-vl" in name or "qwen2_5_vl" in name:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as ModelCls
        else:
            try:
                from transformers import AutoProcessor, Qwen2VLForConditionalGeneration as ModelCls
            except ImportError:
                from transformers import AutoProcessor, AutoModelForImageTextToText as ModelCls

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda",
            "low_cpu_mem_usage": True,
        }
        try:
            self._model = ModelCls.from_pretrained(self.model_name, **kwargs)
        except Exception:
            kwargs["device_map"] = "auto"
            try:
                self._model = ModelCls.from_pretrained(self.model_name, **kwargs)
            except Exception:
                self._model = ModelCls.from_pretrained(
                    self.model_name, trust_remote_code=True, **kwargs
                )
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        self._boot_time = time.time() - t0
        self._parked = False
        print(f"   ✅ VLM loaded in {self._boot_time:,.1f}s")

    def park(self) -> None:
        """Keep weights in RAM, free GPU for LLM / diffusion."""
        if self._model is None or self._parked:
            return
        print("   ♻️  Parking VLM on CPU (no reload from disk)")
        try:
            self._model.to("cpu")
            self._parked = True
        except Exception as exc:
            print(f"⚠️  VLM park skipped ({type(exc).__name__}: {exc})")
            return
        torch.cuda.empty_cache()

    def wake(self) -> None:
        self._ensure_loaded()
        if not self._parked or self._model is None:
            return
        print("   👁️  Waking VLM on CUDA")
        try:
            self._model.to(device="cuda", dtype=torch.bfloat16)
            self._parked = False
        except Exception as exc:
            print(f"⚠️  VLM wake failed ({type(exc).__name__}: {exc}) — leaving as-is")

    def chat_with_image(
        self,
        image_path: str | Path,
        prompt: str,
        system_prompt: str = "You are a precise, honest visual analyst for professional commercial photography. "
        "Describe exactly what you see. If something is unclear, say so. "
        "When asked for JSON, return ONLY valid JSON with no extra commentary.",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._ensure_loaded()
        if hasattr(self, "wake"):
            self.wake()
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"VLM input image not found: {p}")

        image = Image.open(p).convert("RGB")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        temp = self.temperature if temperature is None else temperature
        gen_kw = dict(max_new_tokens=max_tokens or self.max_tokens)
        if temp and temp > 0:
            gen_kw.update(do_sample=True, temperature=temp)
        else:
            gen_kw["do_sample"] = False

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kw)
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def caption(self, image_path: str | Path, max_tokens: int = 120) -> str:
        return self.chat_with_image(
            image_path,
            "Describe this image concisely for a commercial photography catalog. "
            "Mention subject, setting, lighting style, composition, colour palette. 2-3 sentences.",
            max_tokens=max_tokens,
        )
