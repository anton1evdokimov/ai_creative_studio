from pydantic import BaseModel


class ImageGenerationResult(BaseModel):

    image_path: str
    prompt: str
    model: str
    seed: int | None = None