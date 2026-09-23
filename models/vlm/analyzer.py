"""ProductAnalyzer: VLM-based structured analysis of the INPUT product photo.

Reads the raw product image (e.g. serum.jpeg) the user uploaded, asks the VLM
17 detailed questions, then returns a strong ProductAnalysis object used by
all downstream nodes (concept generation, scoring, prompt refinement, etc).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.evaluation import ProductAnalysis
from models.vlm.factory import create_vlm
from models.llm.parser import _safe_json_loads


_PRODUCT_VLM_PROMPT = """Analyze this product photograph carefully. Your goal is to extract every
important visual detail a creative director would need to plan advertising imagery.

Return ONLY a valid JSON object (no markdown fences, no extra prose) with these EXACT keys:
{{
  "category": "e.g. premium cosmetics, beverage, apparel, electronics...",
  "product_type": "e.g. face serum, soda can, t-shirt, wireless earbuds...",
  "product_name": "short commercial name, guess if ambiguous",
  "materials": ["glass", "plastic", "matte paper", "aluminium", ... up to 6 items],
  "colors": ["human color names", ... up to 5],
  "color_palette_hex": ["#AAAAAA", ... 3-5 dominant colours],
  "shape": "2-3 words describing silhouette and packaging geometry",
  "size": "relative size description: palm-sized, travel-size, counter-size...",
  "luxury_level": "ONE of: budget, mid-range, premium, luxury, ultra-luxury",
  "brand_visual_cues": ["logotype style", "graphic marks", "emblems", ... visible on packaging],
  "key_visual_features": ["dropper applicator", "gold foil cap", "rounded shoulder glass", ...],
  "packaging_type": "e.g. glass dropper bottle, blister pack, folding carton, tin can",
  "extracted_text_ocr": "any text you can confidently read on the item (or empty string)",
  "visual_caption": "2-3 sentence detailed commercial description of the photo",
  "suggested_target_audience": "1-2 sentences: who buys this product (demographics + psychographics)"
}}

Think carefully before answering. Be precise and conservative — if something is not legible,
return an empty string/array for that key.
"""


_analyzer_instance = None


def create_product_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ProductAnalyzer()
    return _analyzer_instance


def reset_product_analyzer():
    global _analyzer_instance
    _analyzer_instance = None


class ProductAnalyzer:
    """Use the shared VLM to describe the input product photo -> ProductAnalysis."""

    def __init__(self):
        self.vlm = create_vlm()

    # ------------------------------------------------------------------
    def analyze(
        self,
        image_path: str | Path,
        user_description: str = "",
        fallback_if_missing_image: bool = True,
    ) -> ProductAnalysis:
        p = Path(image_path) if image_path else None

        # Case 1: no image available -> build analysis from text only --------------------
        if p is None or not p.exists():
            if not fallback_if_missing_image and p is not None:
                raise FileNotFoundError(p)
            print(
                f"   ⚠️ Image missing ({image_path!r}) — text-only fallback, "
                "VLM did not see a photo"
            )
            return self._text_only_fallback(user_description or "retail consumer product")

        # Case 2: real image -> ask VLM ---------------------------------------------------
        raw = self.vlm.chat_with_image(
            p,
            _PRODUCT_VLM_PROMPT,
            system_prompt=(
                "You are a senior commercial-studio visual analyst. You must answer ONLY with a "
                "valid JSON object matching the requested schema. Do not wrap in markdown fences, "
                "do not write commentary before or after JSON."
            ),
            max_tokens=900,
            temperature=0.2,
        )
        parsed: dict[str, Any] = _safe_json_loads(raw, {})
        if not isinstance(parsed, dict):
            parsed = {}

        analysis = self._build_typed(parsed, raw)
        if not any((analysis.product_name, analysis.product_type, analysis.visual_caption, analysis.category)):
            snippet = (raw or "").replace("\n", " ")[:280]
            print(f"   ⚠️ VLM JSON empty/unparsed. Raw: {snippet or '∅'}")

        # Merge any user-provided hints (they might know the brand/sku)
        if user_description and not analysis.product_name:
            analysis.product_name = user_description[:120]
        return analysis

    # ------------------------------------------------------------------
    @staticmethod
    def _text_only_fallback(description: str) -> ProductAnalysis:
        """Used when no product image was supplied — infer from user text."""
        desc_lc = description.lower()
        is_cosmetic = any(t in desc_lc for t in ["serum", "cream", "skincare", "lotion", "cosmetic", "beauty"])
        is_premium = any(t in desc_lc for t in ["premium", "luxury", "luxe", "haute", "designer"])
        if is_cosmetic:
            return ProductAnalysis(
                category="premium cosmetics" if is_premium else "beauty & personal care",
                product_type="skincare serum/moisturizer",
                product_name=description[:120],
                materials=["glass bottle", "dropper applicator", "gold foil accents"],
                colors=["golden amber", "transparent", "matte black cap"],
                color_palette_hex=["#E6B325", "#0A0A0A", "#F5EFE1"],
                shape="cylindrical glass bottle, narrow neck",
                size="30 ml hand-sized",
                luxury_level="luxury" if is_premium else "mid-range",
                key_visual_features=["visible golden liquid", "dropper", "glass specular highlights"],
                packaging_type="glass dropper bottle",
                visual_caption=f"Commercial studio photo of {description[:160]}.",
                suggested_target_audience="men/women 25-50 skincare shoppers, mid to high disposable income.",
            )
        return ProductAnalysis(
            category="consumer goods",
            product_type="retail item",
            product_name=description[:120],
            visual_caption=f"Commercial product photograph of: {description[:200]}.",
            suggested_target_audience="general retail shoppers",
            raw_notes="Built from user description, no input image was supplied.",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_typed(parsed: dict[str, Any], raw: str) -> ProductAnalysis:
        def _lst(v, n=8):
            if isinstance(v, list):
                return [str(x)[:120] for x in v if str(x).strip()][:n]
            if isinstance(v, str):
                return [s.strip()[:120] for s in v.split(",") if s.strip()][:n]
            return []

        def _str(v, n=500):
            return (str(v).strip() if v is not None else "")[:n]

        lux = _str(parsed.get("luxury_level"), 40).lower() or "mid-range"
        allowed_lux = {"budget", "mid-range", "premium", "luxury", "ultra-luxury"}
        if lux not in allowed_lux:
            lux = "mid-range"

        return ProductAnalysis(
            category=_str(parsed.get("category"), 160),
            product_type=_str(parsed.get("product_type"), 160),
            product_name=_str(parsed.get("product_name"), 200),
            materials=_lst(parsed.get("materials"), 10),
            colors=_lst(parsed.get("colors"), 10),
            color_palette_hex=_lst(parsed.get("color_palette_hex"), 8),
            shape=_str(parsed.get("shape"), 200),
            size=_str(parsed.get("size"), 120),
            luxury_level=lux,
            brand_visual_cues=_lst(parsed.get("brand_visual_cues"), 12),
            key_visual_features=_lst(parsed.get("key_visual_features"), 14),
            packaging_type=_str(parsed.get("packaging_type"), 160),
            extracted_text_ocr=_str(parsed.get("extracted_text_ocr"), 600),
            visual_caption=_str(parsed.get("visual_caption"), 900),
            suggested_target_audience=_str(parsed.get("suggested_target_audience"), 600),
            raw_notes=(
                _str(parsed.get("raw_notes"), 800) or
                ("VLM raw output:\n" + _str(raw, 4000) if raw else "")
            ),
        )

