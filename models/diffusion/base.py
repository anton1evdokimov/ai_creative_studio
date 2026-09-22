from typing import Any


class ImageBackend:
    """Common interface for all image generation backends (Mock, MLX/mflux, CUDA, etc)."""

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
        **extra: Any,
    ) -> str:
        raise NotImplementedError

