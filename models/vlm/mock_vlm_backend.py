"""Mock VLM backend used when AICS_USE_MOCK=1.

Returns deterministic, plausible answers so the full pipeline runs in <10s without
downloading any vision-model weights. Very useful for LangGraph logic testing.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional


def _stable_rng(seed_text: str) -> random.Random:
    h = hashlib.md5(seed_text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:4], "big"))


_PRESET_PRODUCT_ANALYSIS = {
    "serum": {
        "category": "premium cosmetics",
        "product_type": "face serum",
        "product_name": "vitamin C serum",
        "materials": ["thick clear glass", "rubber dropper bulb", "gold foil cap", "silicone seal"],
        "colors": ["golden amber", "transparent", "matte black cap", "brass accents"],
        "color_palette_hex": ["#E6B325", "#0A0A0A", "#F5EFE1", "#C9A227"],
        "shape": "cylindrical bottle, rounded shoulders, narrow neck",
        "size": "30 ml travel hand size",
        "luxury_level": "luxury",
        "brand_visual_cues": ["no visible logo on this crop", "gold details", "clean silhouette"],
        "key_visual_features": [
            "golden liquid visible through glass",
            "dropper applicator",
            "soft specular highlights on shoulder",
            "high-end weighty glass feel",
        ],
        "packaging_type": "glass dropper bottle",
        "extracted_text_ocr": "no legible text in current crop",
        "visual_caption": (
            "Close-up of a cylindrical 30ml luxury glass dropper bottle containing golden-amber "
            "serum liquid. Soft studio lighting highlights glass shoulders, matte black cap with "
            "gold foil accents creates premium minimalist look."
        ),
        "suggested_target_audience": "women 25-45, high disposable income, luxury skincare buyers",
    },
    "hoodie": {
        "category": "apparel",
        "product_type": "hoodie",
        "product_name": "pullover hoodie",
        "materials": ["cotton fleece", "rib knit cuffs", "drawcord"],
        "colors": ["heather grey", "black"],
        "color_palette_hex": ["#8A8A8A", "#111111", "#F4F4F4"],
        "shape": "oversized pullover, hood, kangaroo pocket",
        "size": "adult regular fit",
        "luxury_level": "mid-range",
        "brand_visual_cues": ["clean chest, no loud logo in crop"],
        "key_visual_features": ["hood", "kangaroo pocket", "rib cuffs", "visible fleece texture"],
        "packaging_type": "folded garment",
        "extracted_text_ocr": "",
        "visual_caption": "Studio photo of a heather grey pullover hoodie with hood and front pocket.",
        "suggested_target_audience": "18-35 streetwear shoppers",
    },
    "default": {
        "category": "consumer goods",
        "product_type": "packaged product",
        "product_name": "retail item",
        "materials": ["cardboard", "plastic", "paper label"],
        "colors": ["white", "brand accent blue", "neutral grey"],
        "color_palette_hex": ["#FFFFFF", "#2E6DB4", "#888888"],
        "shape": "rectangular prism box",
        "size": "palm sized",
        "luxury_level": "mid-range",
        "brand_visual_cues": ["boxed packaging", "printed label"],
        "key_visual_features": ["front label facing camera", "soft shadows below"],
        "packaging_type": "retail box",
        "extracted_text_ocr": "",
        "visual_caption": "Studio shot of a single packaged consumer good on plain background, front on.",
        "suggested_target_audience": "general retail shopper",
    },
}


class MockVLMBackend:
    def __init__(self, latency: float = 0.35):
        print("🧪 Using MockVLMBackend (fast offline synthetic vision answers)")
        self.latency = latency

    def _pick_profile(self, image_path: str, prompt: str) -> dict:
        sig = f"{Path(image_path).name.lower()}::{prompt.lower()}"
        if "hoodie" in sig or "sweatshirt" in sig or "apparel" in sig or "tshirt" in sig or "t-shirt" in sig:
            return dict(_PRESET_PRODUCT_ANALYSIS["hoodie"])
        if "serum" in sig or "cosmetic" in sig or "bottle" in sig:
            return dict(_PRESET_PRODUCT_ANALYSIS["serum"])
        return dict(_PRESET_PRODUCT_ANALYSIS["default"])

    # ------------------------------------------------------------------
    def chat_with_image(
        self,
        image_path: str | Path,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        time.sleep(self.latency)
        p = str(image_path)
        lowered = prompt.lower()

        # --- Image scoring prompt -> return 0..1 scores JSON --------------------
        # NOTE: run this BEFORE product-analysis branch because both can contain the
        # word "json" and scoring is more specific / keyword-rich.
        if any(kw in lowered for kw in [
            "score", "rating", "prompt alignment", "aesthetic", "rate the image",
            "rate on these", "art director", "overall = 0.30", "vlm scoring",
            "6d vlm", "dimensional score", "aesthetic_quality", "product_accuracy",
            "brand_fit", "prompt_alignment",
        ]):
            rng = _stable_rng(p + "::score::" + prompt)
            base_prompt_aligned = 0.82 if "minimalist" in lowered or "studio" in lowered else 0.73
            jitter = lambda v: max(0.0, min(1.0, v + rng.uniform(-0.05, 0.05)))
            pa = jitter(base_prompt_aligned)
            aq = jitter(0.86)
            pr = jitter(0.81)
            re = jitter(0.83)
            bf = jitter(0.79)
            overall = round(0.30 * pa + 0.25 * aq + 0.20 * pr + 0.15 * re + 0.10 * bf, 3)
            return json.dumps(
                {
                    "prompt_alignment": round(pa, 3),
                    "aesthetic_quality": round(aq, 3),
                    "product_accuracy": round(pr, 3),
                    "realism": round(re, 3),
                    "brand_fit": round(bf, 3),
                    "overall": overall,
                    "feedback": (
                        "Сильная сторона: чистое студийное освещение и отделение продукта. "
                        "Слабая сторона: лёгкая diffusion-мягкость по краям."
                    ),
                },
                ensure_ascii=False,
            )

        # --- Product analysis prompt -> return full JSON analysis -----------------
        if any(kw in lowered for kw in [
            "product analysis", "analyze this product", "category:", "materials:",
            "product photograph", "every important visual detail", "creative director",
            "visual_caption", "luxury_level",
        ]):
            profile = self._pick_profile(p, prompt)
            return json.dumps(profile, ensure_ascii=False)

        # --- Fallback caption ----------------------------------------------------
        profile = self._pick_profile(p, prompt)
        return profile.get("visual_caption", "Studio product photograph.")

    def caption(self, image_path: str | Path, max_tokens: int = 120) -> str:
        return self.chat_with_image(image_path, "Describe this image briefly.")[:max_tokens]
