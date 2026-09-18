import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from .base import LLMBackend


class CUDABackend(LLMBackend):

    def __init__(self, model_name):

        self.device = "cuda"

        print(
            f"Loading model on {self.device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )


    def generate(self, prompt: str) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)


        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300
            )


        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        return result