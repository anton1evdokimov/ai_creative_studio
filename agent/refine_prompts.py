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
  "positive_prompt": "A single long string, comma separated, ENGLISH only. You MUST include:
    - EXACT product & its exact visual features (materials, colours, silhouette, cap details, etc) from concept + context
    - camera: e.g. 'Hasselblad X1D II, 80mm f/2.2 lens, leaf shutter, Phase One IQ4 150MP digital back'
    - lighting: TLCI 98+, CRI 95+, 5200K, modifier (large octabox 45deg top-left, white bounce card fill from front-right), hard or soft, light ratio
    - composition: rule-of-thirds, negative space right for copy, product 30% frame, eye-level
    - depth-of-field: f/2.8 focus on the bottle label/cap, softly blurred background
    - background: specific description matching the concept, e.g. raw travertine slab, sun shadow bars
    - style: 'commercial advertising photography, ultra high end retouch, high dynamic range, ultra sharp focus on subject, glossy specular highlights rendered correctly'
    - mood, season, time-of-day if concept specifies them.
    NO numbered lists, NO newlines inside the string.",
  "negative_prompt": "A single string, comma separated, ENGLISH only, fine-tuned for FLUX.1 4-step. Include:
    deformed, malformed, melted glass, broken bottle, extra dropper tubes, 6-finger hand, extra limbs, bad perspective, warped typography, flipped logos, extra reflections,
    deformed text, aliased edges, banding, posterisation, watermarks, captions, jpeg artifacts, over-sharpen halos, floating objects, bad shadows, wrong perspective glass,
    uncanny valley, digital painting, CG render, cartoon, anime, illustration, sketch, lowres, blurry",
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
            negative = (
                "deformed, malformed, melted glass, broken bottle, extra dropper, warped type, "
                "jpeg artifacts, watermarks, floating objects, bad shadows, wrong perspective glass, "
                "CG render, cartoon, illustration, lowres, blurry"
            )
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

        desc = ""
        analysis_dict = None
        if product_analysis is not None:
            try:
                analysis_dict = product_analysis.model_dump()
            except Exception:
                analysis_dict = product_analysis if isinstance(product_analysis, dict) else None
            if isinstance(analysis_dict, dict):
                desc = str(
                    analysis_dict.get("visual_caption")
                    or analysis_dict.get("product_name")
                    or analysis_dict.get("product_type")
                    or ""
                )
        base = concept_to_prompt(concept, desc, analysis_dict)
        return (
            f"{base}, Hasselblad X1D II, 80mm f/2.2 lens, Phase One IQ4 150MP digital back, "
            "TLCI 98+, CRI 95+, 5200K natural daylight, large octabox 45deg top-left, "
            "white bounce card fill from front-right, commercial advertising photography, "
            "ultra high end retouch, ultra sharp focus on product, correct glass reflections, "
            "glossy specular highlights, high dynamic range, print ready"
        )
