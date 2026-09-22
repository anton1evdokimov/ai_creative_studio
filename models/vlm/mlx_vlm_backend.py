"""Real VLM backend via Apple MLX.

Uses Qwen2-VL 2B Instruct 4-bit (the smallest viable vision model that fits 8GB M1).
The model is loaded ONCE and reused across:
  1. Input product analysis (ProductAnalyzer)
  2. Generated-image scoring (image quality + prompt alignment)
  3. Image captioning if needed.

Requires the optional `mlx-vlm` package:
    pip install mlx-vlm
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


class MLXVLMBackend:
    def __init__(self, model_name: str, max_tokens: int = 512, temperature: float = 0.3):
        try:
            from mlx_vlm import load, generate
        except ImportError as e:
            raise RuntimeError(
                "mlx-vlm is not installed. Install it with:  pip install mlx-vlm"
            ) from e

        self._load_fn = load
        self._generate_fn = generate
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._boot_time = None

    @staticmethod
    def _patch_transformers_video_processors() -> None:
        """transformers 5.0.0rc1 + missing torchvision: skip video processor.

        Qwen2-VL image analysis does not need torchvision. We:
        1) prevent `class_name in None` in VIDEO_PROCESSOR_MAPPING_NAMES
        2) load Qwen2VLProcessor from tokenizer + image processor only
        """
        try:
            import importlib

            vpa = importlib.import_module("transformers.models.auto.video_processing_auto")
            names = getattr(vpa, "VIDEO_PROCESSOR_MAPPING_NAMES", None)
            if names:
                for key, value in list(names.items()):
                    if value is None:
                        names[key] = ""
        except Exception:
            pass

        try:
            from transformers import AutoImageProcessor, AutoTokenizer
            from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
            from transformers.processing_utils import ProcessorMixin
        except Exception:
            return

        if not getattr(ProcessorMixin, "_aics_allow_none_video", False):
            original_check = ProcessorMixin.check_argument_for_proper_class

            def _check_argument_for_proper_class(self, argument_name, argument):
                if argument_name == "video_processor" and argument is None:
                    return object
                return original_check(self, argument_name, argument)

            ProcessorMixin.check_argument_for_proper_class = _check_argument_for_proper_class
            ProcessorMixin._aics_allow_none_video = True

        if getattr(Qwen2VLProcessor, "_aics_no_video_patch", False):
            return

        @classmethod
        def from_pretrained_no_video(cls, pretrained_model_name_or_path, *args, **kwargs):
            keep = {
                key: kwargs[key]
                for key in (
                    "trust_remote_code",
                    "revision",
                    "token",
                    "cache_dir",
                    "local_files_only",
                    "use_fast",
                )
                if key in kwargs
            }
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **keep
            )
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **keep
            )
            processor = cls(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=None,
            )
            template = getattr(tokenizer, "chat_template", None)
            if template and not getattr(processor, "chat_template", None):
                processor.chat_template = template
            return processor

        Qwen2VLProcessor.from_pretrained = from_pretrained_no_video
        Qwen2VLProcessor._aics_no_video_patch = True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        t0 = time.time()
        print(f"   👁️  Loading VLM weights: {self.model_name} (first run may download)")
        self._patch_transformers_video_processors()
        self._model, self._processor = self._load_fn(self.model_name)
        self._boot_time = time.time() - t0
        print(f"   ✅ VLM loaded in {self._boot_time:,.1f}s")

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
        """Send one text+image turn to the VLM and return the raw string answer."""
        self._ensure_loaded()
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"VLM input image not found: {p}")

        from mlx_vlm.prompt_utils import apply_chat_template

        config = getattr(self._model, "config", None) or {}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = apply_chat_template(
                self._processor,
                config,
                messages,
                num_images=1,
            )
        except Exception:
            formatted = apply_chat_template(
                self._processor,
                config,
                prompt,
                num_images=1,
            )

        result = self._generate_fn(
            self._model,
            self._processor,
            formatted,
            image=str(p),
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            verbose=False,
        )
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, (list, tuple)) and result:
            return str(result[0])
        return str(result)

    def caption(self, image_path: str | Path, max_tokens: int = 120) -> str:
        return self.chat_with_image(
            image_path,
            "Describe this image concisely for a commercial photography catalog. "
            "Mention subject, setting, lighting style, composition, colour palette. 2-3 sentences.",
            max_tokens=max_tokens,
        )
