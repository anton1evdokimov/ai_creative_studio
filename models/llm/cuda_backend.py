from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBackend


class CUDABackend(LLMBackend):
    def __init__(self, model_name: str, max_tokens: int = 800, temperature: float = 0.7):
        self.device = "cuda"
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self._parked = False
        print(f"Loading model on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def park(self) -> None:
        if self.model is None or self._parked:
            return
        print("   ♻️  Parking LLM on CPU (no reload from disk)")
        try:
            self.model.to("cpu")
            self._parked = True
        except Exception as exc:
            print(f"⚠️  LLM park skipped ({type(exc).__name__}: {exc})")
            return
        torch.cuda.empty_cache()

    def wake(self) -> None:
        if not self._parked or self.model is None:
            return
        print("   🧠 Waking LLM on CUDA")
        try:
            self.model.to(device="cuda", dtype=torch.bfloat16)
            self._parked = False
        except Exception as exc:
            print(f"⚠️  LLM wake failed ({type(exc).__name__}: {exc}) — leaving as-is")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> str:
        self.wake()
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
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
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
