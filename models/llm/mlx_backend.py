from typing import Any

from .base import LLMBackend


class MLXBackend(LLMBackend):

    def __init__(self, name: str, max_tokens: int = 500, temperature: float = 0.7, verbose: bool = False):
        from mlx_lm import load

        print(f"🧠 Loading MLX LLM: {name}")
        self.model, self.tokenizer = load(name)
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.default_verbose = verbose

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a precise, helpful AI assistant for luxury advertising creative direction. Return JSON when asked, no extra commentary.",
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: bool | None = None,
        **extra: Any,
    ) -> str:
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        mt = max_tokens if max_tokens is not None else self.default_max_tokens
        t = temperature if temperature is not None else self.default_temperature
        v = verbose if verbose is not None else self.default_verbose

        kwargs = {
            "prompt": formatted_prompt,
            "max_tokens": mt,
            "verbose": v,
        }
        try:
            from mlx_lm.sample_utils import make_sampler

            kwargs["sampler"] = make_sampler(temp=float(t))
        except Exception:
            kwargs["temperature"] = t

        try:
            response = generate(self.model, self.tokenizer, **kwargs)
        except TypeError:
            kwargs.pop("sampler", None)
            kwargs.pop("temperature", None)
            response = generate(self.model, self.tokenizer, **kwargs)

        return response
