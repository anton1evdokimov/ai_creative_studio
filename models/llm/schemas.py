from pydantic import BaseModel
from typing import List


class CreativeConcept(BaseModel):

    name: str
    scene: str
    lighting: str
    style: str


class CreativeConcepts(BaseModel):

    concepts: List[CreativeConcept]