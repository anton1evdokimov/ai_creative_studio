from typing import List
from typing_extensions import TypedDict

from .creative import CreativeConcept
from .evaluation import ImageEvaluation


class PipelineState(TypedDict):

    product_description: str

    creative_concepts: List[CreativeConcept]

    generated_images: List[str]

    evaluation_results: List[ImageEvaluation]