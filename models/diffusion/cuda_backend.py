from pathlib import Path

import torch

from .base import ImageBackend
from .config import resolve_hf_model


class FluxCUDABackend(ImageBackend):

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        from diffusers import FluxPipeline

        model_id = resolve_hf_model(self.config["model"])
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Loading FLUX.1 ({model_id}) via Diffusers on {self.device}")

        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        return pipe

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
        **extra,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)

        pipe_kwargs = dict(
            prompt=prompt,
            height=self.config["height"],
            width=self.config["width"],
            guidance_scale=self.config["guidance_scale"],
            num_inference_steps=self.config["num_inference_steps"],
            max_sequence_length=512,
            generator=generator,
        )
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        try:
            image = self.pipe(**pipe_kwargs).images[0]
        except TypeError:
            pipe_kwargs.pop("negative_prompt", None)
            image = self.pipe(**pipe_kwargs).images[0]

        image.save(output_path)
        return output_path
