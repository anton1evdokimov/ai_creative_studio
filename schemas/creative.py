from pydantic import BaseModel
from typing import List, Optional


class CreativeConcept(BaseModel):
    name: str
    scene: str
    lighting: str
    style: str
    mood: Optional[str] = ""
    camera_angle: Optional[str] = ""
    color_palette: Optional[List[str]] = []
    composition: Optional[str] = ""
    tags: Optional[List[str]] = []


class ConceptScore(BaseModel):
    creativity: float
    relevance: float
    realizability: float
    prompt_quality: float
    avg: float
    feedback: str = ""


class ScoredConcept(BaseModel):
    concept: CreativeConcept
    score: ConceptScore


class CreativeConcepts(BaseModel):
    concepts: List[CreativeConcept]