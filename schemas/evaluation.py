from pydantic import BaseModel


class ImageEvaluation(BaseModel):

    image_path: str
    clip_score: float
    quality_score: float