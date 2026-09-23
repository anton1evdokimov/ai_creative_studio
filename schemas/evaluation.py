from typing import Optional

from pydantic import BaseModel


class ProductAnalysis(BaseModel):
    """Structured output from VLM analyzing the input product photograph."""
    category: str = ""
    product_type: str = ""
    product_name: str = ""
    materials: list[str] = []
    colors: list[str] = []
    color_palette_hex: list[str] = []
    shape: str = ""
    size: str = ""
    luxury_level: str = ""  # budget / mid-range / premium / luxury / ultra-luxury
    brand_visual_cues: list[str] = []
    key_visual_features: list[str] = []
    packaging_type: str = ""
    extracted_text_ocr: str = ""
    visual_caption: str = ""
    suggested_target_audience: str = ""
    raw_notes: str = ""


class VLMImageScores(BaseModel):
    """Scores for a generated image (all 0..1).
    Using VLM-as-a-judge instead of a separate CLIP model — saves 1-2 GB on 8GB M1."""
    prompt_alignment: float = 0.0  # how well image matches the prompt
    aesthetic_quality: float = 0.0     # composition, lighting, overall beauty
    product_accuracy: float = 0.0      # does product look correct, no weird artifacts
    realism: float = 0.0            # photorealism
    brand_fit: float = 0.0         # fits luxury / product category vibes
    overall: float = 0.0             # weighted avg
    feedback: str = ""


class CLIPScores(BaseModel):
    """Deterministic CLIP metrics (cosines + LAION aesthetic)."""
    clip_t: float = 0.0          # prompt alignment, 0..1 (CLIPScore-style)
    clip_t_raw: float = 0.0      # raw cosine
    clip_i: Optional[float] = None  # gen vs product photo, 0..1
    clip_i_raw: Optional[float] = None
    aesthetic: float = 0.0       # 0..1 (raw/10)
    aesthetic_raw: float = 0.0   # typical 1..10 LAION scale


class ImageEvaluation(BaseModel):
    image_path: str
    prompt: str = ""
    clip_score: Optional[float] = None  # CLIP-T 0..1
    quality_score: Optional[float] = None  # heuristics 0..1
    clip_metrics: Optional[CLIPScores] = None
    vlm_scores: Optional[VLMImageScores] = None
    score: float = 0.0  # final blended score used by router