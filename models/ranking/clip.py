"""Image scorer/ranker for generated images.

INTENTIONAL DESIGN for 8GB M1 RAM:
    We do NOT load a separate CLIP model (would consume another 600MB-1.5GB GPU RAM and
    force the main LLM/VLM to get partially unloaded on 8GB -> very slow).

    Instead we use the SAME shared Qwen2-VL-2B VLM instance as:
      1. `prompt_alignment`  - does the image match the diffusion prompt? (replaces CLIP cos-sim)
      2. `aesthetic_quality` - composition, lighting, overall beauty
      3. `product_accuracy`  - product shape/materials/artifacts (no 6-finger droppers etc!)
      4. `realism`           - photorealism vs uncanny diffusion look
      5. `brand_fit`         - fits the luxury / category / target audience vibe

If later you have more RAM you can add a real clip-vit-base in a separate class here and
keep the same public API.
"""
from __future__ import annotations

import json
from pathlib import Path

from schemas.evaluation import VLMImageScores, ImageEvaluation
from models.vlm.factory import create_vlm
from models.llm.parser import _safe_json_loads


_IMAGE_SCORING_PROMPT = """You are a senior commercial photography art director rating a
candidate generated ad image. Answer ONLY with a valid JSON object. No markdown fences,
no extra words.

Reference diffusion prompt:
"{prompt_text}"

Product / brand context:
"{context_text}"

Rate the image on these 5 dimensions using a score from 0.00 (terrible) to 1.00 (perfect, print-ready):
  - prompt_alignment:  how faithfully does the image depict the prompt above? no missing key elements.
  - aesthetic_quality:  composition, lighting quality, colour harmony, overall beauty.
  - product_accuracy:   the hero product has no diffusion artifacts. Correct proportions, materials,
                        no extra limbs / melted parts.
  - realism:            photorealism: believable lighting physics, plausible textures, not uncanny.
  - brand_fit:          matches the luxury / product-category vibe from context.

Also compute "overall" = 0.30*prompt_alignment + 0.25*aesthetic_quality + 0.20*product_accuracy
                        + 0.15*realism + 0.10*brand_fit.

Also include "feedback": 1–2 short sentences in Russian (Cyrillic). Explain the single
strongest strength and the single strongest weakness. Do not write feedback in English.

Return JSON:
{{
  "prompt_alignment": 0.00,
  "aesthetic_quality": 0.00,
  "product_accuracy": 0.00,
  "realism": 0.00,
  "brand_fit": 0.00,
  "overall": 0.00,
  "feedback": "Сильная сторона: … Слабая сторона: …"
}}
"""


_scorer_instance = None


def create_image_scorer():
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = VLMImageScorer()
    return _scorer_instance


def reset_image_scorer():
    global _scorer_instance
    _scorer_instance = None


class VLMImageScorer:
    def __init__(self):
        self.vlm = create_vlm()

    # ------------------------------------------------------------------
    def score(
        self,
        image_path: str | Path,
        prompt_text: str,
        context_text: str = "luxury skincare product advertisement",
    ) -> VLMImageScores:
        p = Path(image_path)
        if not p.exists():
            return VLMImageScores(feedback="Файл изображения не найден, оценка обнулена.")

        filled_prompt = _IMAGE_SCORING_PROMPT.format(
            prompt_text=prompt_text[:900] if prompt_text else "",
            context_text=context_text[:700] if context_text else "",
        )
        raw = self.vlm.chat_with_image(
            p,
            filled_prompt,
            system_prompt=(
                "You are a critical but fair commercial art director. "
                "Respond ONLY with valid JSON, never with commentary. "
                "The numeric keys stay in English. The feedback string MUST be Russian."
            ),
            max_tokens=500,
            temperature=0.2,
        )
        parsed = _safe_json_loads(raw, {})
        if not isinstance(parsed, dict):
            parsed = {}

        def _f(key: str, default: float = 0.0) -> float:
            try:
                v = float(parsed.get(key, default))
            except Exception:
                v = default
            return max(0.0, min(1.0, v))

        prompt_alignment = _f("prompt_alignment")
        aesthetic_quality = _f("aesthetic_quality")
        product_accuracy = _f("product_accuracy")
        realism = _f("realism")
        brand_fit = _f("brand_fit")
        explicit_overall = _f("overall")
        computed = round(
            0.30 * prompt_alignment + 0.25 * aesthetic_quality + 0.20 * product_accuracy
            + 0.15 * realism + 0.10 * brand_fit,
            3,
        )
        overall = explicit_overall if parsed.get("overall") not in (None, 0, 0.0) else computed
        feedback = str(parsed.get("feedback") or "")[:600]

        return VLMImageScores(
            prompt_alignment=round(prompt_alignment, 3),
            aesthetic_quality=round(aesthetic_quality, 3),
            product_accuracy=round(product_accuracy, 3),
            realism=round(realism, 3),
            brand_fit=round(brand_fit, 3),
            overall=round(overall, 3),
            feedback=feedback,
        )

    # ------------------------------------------------------------------
    def score_image_result(
        self,
        image_path: str | Path,
        prompt_text: str,
        context_text: str = "",
    ) -> ImageEvaluation:
        scores = self.score(image_path, prompt_text=prompt_text, context_text=context_text)
        return ImageEvaluation(
            image_path=str(image_path),
            prompt=prompt_text,
            clip_score=scores.prompt_alignment,   # backwards-compat alias (CLIP replacement)
            quality_score=scores.aesthetic_quality,  # backwards-compat alias
            vlm_scores=scores,
            score=scores.overall,
        )
