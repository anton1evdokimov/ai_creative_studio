from pathlib import Path
from typing import Optional

from schemas.creative import CreativeConcept
from schemas.generation import ImageGenerationResult, RefinedPrompt

from .config import load_diffusion_config
from .factory import create_image_backend


from models.product_kind import ensure_product_lead, looks_like_apparel, product_noun


def concept_to_prompt(
    concept: CreativeConcept,
    product_description: str,
    product_analysis: Optional[dict] = None,
) -> str:
    """Legacy naive prompt builder used ONLY when no refined_prompt is supplied."""
    description = product_noun(product_analysis, product_description)
    apparel = looks_like_apparel(product_analysis, description)
    finish = (
        "Photorealistic fashion photo, ultra detailed fabric, sharp stitching, magazine lookbook."
        if apparel
        else "Photorealistic product shot, ultra detailed, 8k, magazine quality, sharp focus."
    )
    parts = [
        f"Professional advertising photography of {description}.",
    ]
    if concept.scene:
        parts.append(f"Scene: {concept.scene}.")
    if concept.lighting:
        parts.append(f"Lighting: {concept.lighting}.")
    if concept.style:
        parts.append(f"Style: {concept.style}.")
    if concept.mood:
        parts.append(f"Mood: {concept.mood}.")
    if concept.camera_angle:
        parts.append(f"Camera: {concept.camera_angle}.")
    if concept.composition:
        parts.append(f"Composition: {concept.composition}.")
    if concept.color_palette:
        parts.append("Color palette: " + ", ".join(str(c) for c in concept.color_palette) + ".")
    if concept.tags:
        parts.append(", ".join(str(t) for t in concept.tags) + ".")
    parts.append(finish)
    return " ".join(parts)


_flux_generator = None


def get_flux_generator() -> "FluxGenerator":
    global _flux_generator
    if _flux_generator is None:
        _flux_generator = FluxGenerator()
    return _flux_generator


def reset_flux_generator():
    global _flux_generator
    _flux_generator = None


class FluxGenerator:
    def __init__(self):
        self.config = load_diffusion_config()
        self.backend = create_image_backend()

    def _find_refined(
        self,
        index: int,
        concept: CreativeConcept,
        refined_prompts: Optional[list[RefinedPrompt]],
    ) -> Optional[RefinedPrompt]:
        if not refined_prompts:
            return None
        if index < len(refined_prompts):
            rp = refined_prompts[index]
            # name sanity check — if index is clearly shuffled, fall back to name match
            if (
                rp.concept_name == concept.name
                or (rp.concept and rp.concept.name == concept.name)
            ):
                return rp
            # Otherwise try to find by name in the list
            for rp in refined_prompts:
                if rp.concept_name == concept.name:
                    return rp
        return None

    def generate_concepts(
        self,
        concepts: list[CreativeConcept],
        product_description: str,
        refined_prompts: Optional[list[RefinedPrompt]] = None,
        product_analysis: Optional[dict] = None,
        product_image: str = "",
    ) -> list[ImageGenerationResult]:
        results = []
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, concept in enumerate(concepts):
            rp = self._find_refined(index, concept, refined_prompts)
            if rp is not None:
                prompt = rp.positive_prompt
                if rp.style_boost_tags:
                    # append, de-dupe but keep order
                    seen = set(prompt.lower().split(", "))
                    extras = ", ".join(t for t in rp.style_boost_tags if t.lower() not in seen)
                    if extras:
                        prompt = prompt.rstrip().rstrip(",") + ", " + extras
            else:
                prompt = concept_to_prompt(concept, product_description, product_analysis)

            noun = product_noun(product_analysis, product_description, product_image)
            prompt = ensure_product_lead(prompt, noun)

            slug = concept.name.lower().replace(" ", "_") or f"concept_{index}"
            slug = "".join(ch for ch in slug if ch.isalnum() or ch in "_-")[:60] or f"concept_{index}"
            output_path = str(output_dir / f"{index:02d}_{slug}.png")

            print(f"🎨 Generating [{index+1}/{len(concepts)}]: {concept.name}")
            if rp is not None:
                print(f"   (Refined prompt, length {len(prompt)})")
                print(f"   Preview: {prompt[:120]}…")
                if rp.negative_prompt:
                    print(f"   Neg:     {rp.negative_prompt[:80]}…")
            else:
                print(f"   (Naive prompt, length {len(prompt)})")
                print(f"   Preview: {prompt[:120]}…")

            image_path = self.backend.generate(
                prompt=prompt,
                output_path=output_path,
                seed=index,
                negative_prompt=rp.negative_prompt if rp else None,
                control_image=product_image,
                ip_adapter_image=product_image,
            )

            results.append(
                ImageGenerationResult(
                    image_path=image_path,
                    prompt=prompt,
                    refined_prompt=rp,
                    model=str(self.config["model"]),
                    seed=index,
                )
            )

        return results

