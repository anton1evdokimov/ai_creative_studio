from pathlib import Path

from .base import ImageBackend


class FluxMLXBackend(ImageBackend):

    def __init__(self, config: dict):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        model_name = self.config["model"]
        quantize = self.config["quantize"]

        try:
            from mflux.models.flux.variants.txt2img.flux import Flux1
        except ImportError:
            try:
                from mflux.flux.flux import Flux1
            except ImportError as error:
                raise RuntimeError(
                    "FLUX.1 on Mac needs mflux. Install with: pip install mflux"
                ) from error

        print(f"Loading FLUX.1 ({model_name}) via MLX / mflux")

        return Flux1.from_name(
            model_name=model_name,
            quantize=quantize,
        )

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
        **extra,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if negative_prompt:
            print("   (schnell/MLX ignores negative_prompt)")

        image = self._run(
            prompt=prompt,
            seed=0 if seed is None else seed,
        )

        save = getattr(image, "save")
        try:
            save(path=output_path)
        except TypeError:
            save(output_path)

        return output_path

    def _run(self, prompt: str, seed: int):
        steps = self.config["num_inference_steps"]
        height = self.config["height"]
        width = self.config["width"]

        try:
            return self.model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )
        except TypeError:
            from mflux.config.config import Config

            return self.model.generate_image(
                seed=seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                ),
            )
