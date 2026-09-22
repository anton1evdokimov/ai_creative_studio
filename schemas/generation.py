from pydantic import BaseModel

from schemas.creative import CreativeConcept


class RefinedPrompt(BaseModel):
    """Output of the prompt-refinement node: polished diffusion prompts for FLUX.1."""
    concept: CreativeConcept
    concept_name: str
    positive_prompt: str
    negative_prompt: str
    style_boost_tags: list[str] = []
    estimated_prompt_strength_notes: str = ""
    raw_llm_output: str = ""


class ImageGenerationResult(BaseModel):
    image_path: str
    prompt: str
    refined_prompt: RefinedPrompt | None = None
    model: str
    seed: int | None = None