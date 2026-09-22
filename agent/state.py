from typing import TypedDict, List, Dict, Any

from schemas.creative import CreativeConcept, ScoredConcept
from schemas.evaluation import ProductAnalysis, ImageEvaluation
from schemas.generation import RefinedPrompt, ImageGenerationResult


class PipelineState(TypedDict, total=False):
    """LangGraph pipeline state (all fields optional so partial-node outputs are fine).
    Keeps BOTH legacy Dict views AND new typed Pydantic objects so existing consumers work.
    """

    # INPUTS
    product_image: str
    product_description: str

    # ---- STAGE 1: ANALYSIS (VLM) ----
    product_analysis: Dict[str, Any]        # legacy dict view
    product_analysis_obj: ProductAnalysis   # new typed VLM analysis result (17 fields)

    # ---- STAGE 2+3: CONCEPTION + SCORING ----
    creative_concepts: List[CreativeConcept]
    scored_concepts: List[ScoredConcept]    # all, sorted by avg
    ranked_concepts: List[CreativeConcept]  # top-K winners

    # ---- STAGE 3b: PROMPT REFINEMENT (NEW: professional FLUX.1 prompts) ----
    refined_prompts: List[RefinedPrompt]

    # ---- STAGE 4: GENERATION ----
    generation_results: List[ImageGenerationResult]
    generated_images: List[str]             # legacy list of paths
    generated_videos: List[str]

    # ---- STAGE 5: EVALUATION ----
    evaluation_results: List[Dict[str, Any]]   # legacy dict view
    evaluation_typed: List[ImageEvaluation]    # new 6D VLM + quality typed view
    best_image: str

    # ---- CONTROL FLOW ----
    retry_count: int
