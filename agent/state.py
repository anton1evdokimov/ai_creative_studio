from typing import TypedDict, List, Dict, Any

from schemas.creative import CreativeConcept


class PipelineState(TypedDict):

    # входные данные
    product_description: str

    # результат анализа товара
    product_analysis: Dict[str, Any]

    # идеи от LLM
    creative_concepts: List[CreativeConcept]

    # будущие изображения
    generated_images: List[str]
    
    generated_videos: List[str]

    # результаты оценки
    evaluation_results: Dict[str, Any]
