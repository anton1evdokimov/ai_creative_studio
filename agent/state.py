from typing import TypedDict, List
from models.llm.schemas import CreativeConcept


class CreativeState(TypedDict):
      # input
    product_image: str
    product_description: str

    # analysis
    product_analysis: dict

    # planning
    creative_concepts: list[CreativeConcept]

    # generation
    generated_images: List[str]

    # evaluation
    evaluation_results: List[dict]

    # output
    best_image: str
