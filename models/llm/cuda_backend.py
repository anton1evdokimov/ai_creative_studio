from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBackend


class CUDABackend(LLMBackend):
    def __init__(self, model_name: str, max_tokens: int = 800, temperature: float = 0.7):
        self.device = "cuda"
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        print(f"Loading model on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        mt = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        temp = float(temperature if temperature is not None else self.default_temperature)
        gen_kw: dict = dict(max_new_tokens=mt)
        if temp > 0:
            gen_kw.update(do_sample=True, temperature=temp)
        else:
            gen_kw["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kw)

        trimmed = outputs[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
