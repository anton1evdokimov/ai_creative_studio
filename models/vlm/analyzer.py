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
  "category": "e.g. apparel, premium cosmetics, beverage, electronics",
  "product_type": "e.g. hoodie, t-shirt, face serum",
  "product_name": "short commercial name",
  "materials": ["cotton fleece", "... max 4"],
  "colors": ["pink", "... max 4"],
  "color_palette_hex": ["#E8A0B0", "... max 3"],
  "shape": "short silhouette",
  "size": "relative size",
  "luxury_level": "ONE of: budget, mid-range, premium, luxury, ultra-luxury",
  "brand_visual_cues": ["... max 4"],
  "key_visual_features": ["hood", "kangaroo pocket", "... max 5"],
  "packaging_type": "garment / hangtag / bottle / box",
  "extracted_text_ocr": "line1\\nline2\\nline3 — ALL visible label lines, keep order top-to-bottom",
  "visual_caption": "ONE short sentence",
  "suggested_target_audience": "short phrase"
}}

Keep arrays short. JSON MUST be fully closed. If clothing: category=apparel, fabrics not glass.
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
            hint = ""
            try:
                from models.product_kind import filename_hint
                hint = filename_hint(p) if p else ""
            except Exception:
                hint = ""
            return self._text_only_fallback(user_description or hint or "retail consumer product")

        # Case 2: real image -> ask VLM ---------------------------------------------------
        raw = self.vlm.chat_with_image(
            p,
            _PRODUCT_VLM_PROMPT,
            system_prompt=(
                "You are a senior commercial-studio visual analyst. You must answer ONLY with a "
                "valid JSON object matching the requested schema. Do not wrap in markdown fences, "
                "do not write commentary before or after JSON."
            ),
            max_tokens=1200,
            temperature=0.2,
        )
        parsed: dict[str, Any] = _safe_json_loads(raw, {})
        if not isinstance(parsed, dict):
            parsed = {}

        analysis = self._build_typed(parsed, raw)
        if not any((analysis.product_name, analysis.product_type, analysis.visual_caption, analysis.category)):
            snippet = (raw or "").replace("\n", " ")[:280]
            print(f"   ⚠️ VLM JSON empty/unparsed. Raw: {snippet or '∅'}")

        analysis = self._enrich_from_hints(analysis, p, user_description)
        analysis = self._retry_caption(analysis, p)
        return analysis

    # ------------------------------------------------------------------
    @staticmethod
    def _text_only_fallback(description: str) -> ProductAnalysis:
        """Used when no product image was supplied — infer from user text."""
        from models.product_kind import looks_like_apparel

        desc_lc = description.lower()
        is_cosmetic = any(t in desc_lc for t in ["serum", "cream", "skincare", "lotion", "cosmetic", "beauty"])
        is_premium = any(t in desc_lc for t in ["premium", "luxury", "luxe", "haute", "designer"])
        if looks_like_apparel(description=description):
            return ProductAnalysis(
                category="apparel",
                product_type=description[:80] or "garment",
                product_name=description[:120],
                materials=["cotton fleece", "rib knit cuffs", "metal zipper hardware"],
                colors=["heather grey", "black"],
                color_palette_hex=["#8A8A8A", "#111111", "#F4F4F4"],
                shape="pullover garment silhouette, hood and torso",
                size="adult regular fit",
                luxury_level="premium" if is_premium else "mid-range",
                key_visual_features=["hood", "kangaroo pocket", "rib cuffs", "visible fabric texture"],
                packaging_type="folded garment / hangtag",
                visual_caption=f"Studio fashion photo of {description[:160]}.",
                suggested_target_audience="18-35 streetwear and casual apparel shoppers.",
            )
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

        ocr = parsed.get("extracted_text_ocr")
        if isinstance(ocr, list):
            ocr = "\n".join(str(x).strip() for x in ocr if str(x).strip())

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
            extracted_text_ocr=_str(ocr, 600),
            visual_caption=_str(parsed.get("visual_caption"), 900),
            suggested_target_audience=_str(parsed.get("suggested_target_audience"), 600),
            raw_notes=(
                _str(parsed.get("raw_notes"), 800) or
                ("VLM raw output:\n" + _str(raw, 4000) if raw else "")
            ),
        )

    @staticmethod
    def _enrich_from_hints(analysis: ProductAnalysis, image_path: Path | None, user_description: str) -> ProductAnalysis:
        from models.product_kind import filename_hint, looks_like_apparel, product_noun

        hint = filename_hint(image_path or "")
        apparel = looks_like_apparel(analysis, description=user_description, image_path=image_path or "")
        if apparel and (not analysis.category or "cosmetic" in analysis.category.lower()):
            analysis.category = "apparel"
        if not analysis.product_type and hint:
            analysis.product_type = hint
        if not analysis.product_name:
            analysis.product_name = (user_description[:120] if user_description else "") or hint or analysis.product_type
        if not analysis.visual_caption and (analysis.product_type or hint):
            noun = product_noun(analysis, user_description, image_path or "")
            analysis.visual_caption = f"Commercial photo of {noun}."
        return analysis

    def _retry_caption(self, analysis: ProductAnalysis, image_path: Path) -> ProductAnalysis:
        from models.diffusion.config import load_vlm_config

        cfg = load_vlm_config()
        metric = str(cfg.get("caption_metric") or "clip").lower()
        threshold = float(cfg.get("caption_score_threshold") or 0.35)
        max_retries = max(0, int(cfg.get("caption_max_retries") or 0))
        caption = (analysis.visual_caption or "").strip()
        clipper = None
        try:
            for attempt in range(max_retries + 1):
                vlm_score, suggested = self._judge_caption(image_path, caption)
                auto = None
                if metric == "clip":
                    try:
                        auto, clipper = self._clip_caption_score(image_path, caption, clipper)
                    except Exception as exc:
                        print(f"   caption CLIP skipped ({type(exc).__name__}: {exc})")
                        auto = None
                elif metric == "dino":
                    print("   caption metric=dino skipped (DINOv2 has no text encoder); VLM score only")
                else:
                    print(f"   caption metric={metric!r} unknown; VLM score only")
                auto_ok = auto is None or auto >= threshold
                ok = vlm_score >= threshold and auto_ok
                auto_txt = f"{auto:.3f}" if auto is not None else "—"
                print(
                    f"   caption try {attempt + 1}/{max_retries + 1} "
                    f"vlm={vlm_score:.3f} {metric}={auto_txt} thr={threshold:.2f} "
                    f"{'ok' if ok else 'retry'}: {caption[:120]}"
                )
                if ok or attempt >= max_retries:
                    if suggested and not ok:
                        caption = suggested
                    break
                caption = suggested if suggested and suggested != caption else self._rewrite_caption(image_path, caption)
        finally:
            if clipper is not None:
                from models.ranking.clip_metrics import unload_clip_metrics

                unload_clip_metrics()
        analysis.visual_caption = caption
        return analysis

    def _judge_caption(self, image_path: Path, caption: str) -> tuple[float, str]:
        raw = self.vlm.chat_with_image(
            image_path,
            (
                "Caption to check:\n"
                f"{caption or '(empty)'}\n\n"
                "Look at the photo. Score how well this ONE caption matches visible objects, "
                "colors, packaging and label text. Do not reward invented details.\n"
                'Return ONLY JSON: {"score": 0.0, "caption": "one accurate sentence"}\n'
                "score is 0..1. If the caption is already right, repeat it."
            ),
            system_prompt="You check image captions. Reply with JSON only.",
            max_tokens=300,
            temperature=0.2,
        )
        parsed = _safe_json_loads(raw, {})
        if not isinstance(parsed, dict):
            parsed = {}
        try:
            score = float(parsed.get("score"))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        suggested = str(parsed.get("caption") or "").strip()
        return score, suggested

    def _rewrite_caption(self, image_path: Path, caption: str) -> str:
        raw = self.vlm.chat_with_image(
            image_path,
            (
                "The previous caption was weak:\n"
                f"{caption or '(empty)'}\n"
                'Write a better one. Return ONLY JSON: {"caption": "one sentence"}'
            ),
            system_prompt="You write accurate product-photo captions. Reply with JSON only.",
            max_tokens=200,
            temperature=0.3,
        )
        parsed = _safe_json_loads(raw, {})
        text = str(parsed.get("caption") or "").strip() if isinstance(parsed, dict) else ""
        return text or caption

    def _clip_caption_score(self, image_path: Path, caption: str, clipper):
        from models.diffusion.config import load_ranking_config
        from models.ranking.clip_metrics import create_clip_metrics

        if clipper is None:
            rank = load_ranking_config()
            clipper = create_clip_metrics(rank.get("clip_model") or "openai/clip-vit-large-patch14", use_aesthetic=False)
        scored = clipper.score(image_path, caption or "a product photograph")
        return float(scored.clip_t), clipper

