"""Prompt Refiner node: converts a CreativeConcept into a professional FLUX.1 diffusion prompt.

The LLM rewrites the concept into:
  - A positive_prompt rich with:
      * camera/lens (Hasselblad X1D, 80mm f/2.2, Phase One IQ4 150MP)
      * CRI / colour temperature (TLCI 98+, 5200K)
      * lighting direction, modifiers (octabox, strip, bounce)
      * materials micro-description (brushed brass, soda-lime glass, anodised)
      * 1-2 strong style tags (editorial, commercial, minimalist, etc.)
      * brand/category-specific aesthetic language (luxury cosmetics / apothecary / etc)
  - A negative_prompt tuned for FLUX.1 to remove diffusion artifacts.
  - Extra style-boost tags the caller can optionally append.

Output is always JSON parsed via the same robust parser used everywhere.
"""
from __future__ import annotations

import json
import time
from typing import Any

from schemas.creative import CreativeConcept
from schemas.generation import RefinedPrompt
from models.llm.factory import create_llm
from models.llm.parser import _safe_json_loads


_REFINEMENT_PROMPT = """You are a senior diffusion prompt engineer for commercial advertising photography,
specialising in FLUX.1-schnell 4-step sampling on Apple Silicon. Given a creative concept for a product
advertisement, produce a polished, highly specific prompt set.

## Product / campaign context (VLM analysis of the real product)
```
{product_context}
```

## Creative concept to translate
```json
{concept_json}
```

## What to output: a single JSON object
```
{{
  "positive_prompt": "A single long string, comma separated, ENGLISH only. START with the exact product
    (hoodie / t-shirt / serum bottle / etc) from context — never leave the subject unnamed.
    You MUST include:
    - EXACT product and its visual features from concept + context (if apparel: fabric, colorway, hood/pockets/print/fit;
      if cosmetics: materials, cap, liquid, silhouette)
    - camera: e.g. 'Hasselblad X1D II, 80mm f/2.2 lens'
    - lighting: TLCI 98+, modifier, direction
    - composition matching the concept
    - background matching the concept (apparel: cyclorama / street wall / hanger — NOT a bottle pedestal unless it is cosmetics)
    - style: commercial advertising or fashion lookbook photography, ultra sharp on the product
    NO numbered lists, NO newlines inside the string.",
  "negative_prompt": "A single string, comma separated, ENGLISH only. If apparel: extra hoods, extra sleeves, melted fabric, deformed collar, extra limbs.
    If bottled cosmetics: melted glass, broken bottle, extra dropper. Also: watermarks, captions, jpeg artifacts, cartoon, blurry.",
  "style_boost_tags": ["list", "of", "2-6", "extra", "tags"],
  "estimated_prompt_strength_notes": "1 short sentence explaining how likely this prompt is to produce on-model luxury results for FLUX.1 schnell 4-step"
}}
```

Answer ONLY the JSON object, no markdown fences, no preamble, no commentary."""


_refiner_instance = None


def create_prompt_refiner():
    global _refiner_instance
    if _refiner_instance is None:
        _refiner_instance = PromptRefiner()
    return _refiner_instance


def reset_prompt_refiner():
    global _refiner_instance
    _refiner_instance = None


class PromptRefiner:
    def __init__(self):
        self.llm = create_llm()

    # ------------------------------------------------------------------
    def refine(
        self,
        concept: CreativeConcept,
        product_analysis: Any | None = None,
    ) -> RefinedPrompt:
        t0 = time.time()
        context_text = self._format_context(product_analysis)
        concept_json = concept.model_dump_json(indent=2)

        filled = _REFINEMENT_PROMPT.format(
            product_context=context_text[:2200] if context_text else "(no product image provided, use concept fields only)",
            concept_json=concept_json,
        )
        raw = self.llm.generate(
            filled,
            system_prompt=(
                "You are a precision FLUX prompt engineer. Answer ONLY a valid JSON object. "
                "No markdown fences, no prose before or after."
            ),
            max_tokens=900,
            temperature=0.45,
        )
        parsed = _safe_json_loads(raw, {})
        if not isinstance(parsed, dict):
            parsed = {}

        positive = self._as_str(parsed.get("positive_prompt"))
        negative = self._as_str(parsed.get("negative_prompt"))
        tags = self._as_list(parsed.get("style_boost_tags"))
        notes = self._as_str(parsed.get("estimated_prompt_strength_notes"))

        if not positive or self._looks_like_instruction(positive):
            positive = self._fallback_positive(concept, product_analysis)
        if not negative or self._looks_like_instruction(negative):
            from models.product_kind import category_negative, looks_like_apparel

            negative = category_negative(looks_like_apparel(product_analysis))
        took = time.time() - t0

        return RefinedPrompt(
            concept=concept,
            concept_name=concept.name,
            positive_prompt=positive.strip().rstrip(","),
            negative_prompt=negative.strip().rstrip(","),
            style_boost_tags=tags,
            estimated_prompt_strength_notes=notes or f"Auto-fallback prompt. LLM refine took {took:,.1f}s.",
            raw_llm_output=raw[:4000] if raw else "",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_instruction(text: str) -> bool:
        low = (text or "").lower()
        return any(
            marker in low
            for marker in (
                "you must include",
                "a single long string",
                "comma separated, english only",
                "answer only the json",
                "fine-tuned for flux.1",
            )
        )

    @staticmethod
    def _as_str(v) -> str:
        if isinstance(v, str):
            return v
        return (str(v) if v is not None else "").strip()

    @staticmethod
    def _as_list(v, n: int = 10) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip()[:80] for x in v if str(x).strip()][:n]
        if isinstance(v, str):
            return [s.strip()[:80] for s in v.split(",") if s.strip()][:n]
        return []

    # ------------------------------------------------------------------
    @staticmethod
    def _format_context(product_analysis) -> str:
        if product_analysis is None:
            return ""
        try:
            data = product_analysis.model_dump()
        except Exception:
            data = product_analysis if isinstance(product_analysis, dict) else {}
        lines = []
        for k, v in data.items():
            if not v:
                continue
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v if x)
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_positive(concept: CreativeConcept, product_analysis) -> str:
        from models.diffusion.generator import concept_to_prompt
        from models.product_kind import ensure_product_lead, looks_like_apparel, product_noun

        desc = product_noun(product_analysis)
        analysis_dict = None
        if product_analysis is not None:
            try:
                analysis_dict = product_analysis.model_dump()
            except Exception:
                analysis_dict = product_analysis if isinstance(product_analysis, dict) else None
            if isinstance(analysis_dict, dict):
                desc = product_noun(analysis_dict)
        base = concept_to_prompt(concept, desc, analysis_dict)
        base = ensure_product_lead(base, desc)
        apparel = looks_like_apparel(product_analysis, desc)
        extra = (
            "fashion lookbook photography, sharp fabric texture, visible stitching, natural drape"
            if apparel
            else "correct glass reflections, glossy specular highlights"
        )
        return (
            f"{base}, Hasselblad X1D II, 80mm f/2.2 lens, "
            "TLCI 98+, CRI 95+, 5200K, large octabox 45deg top-left, "
            f"commercial advertising photography, ultra sharp focus on product, {extra}"
        )
