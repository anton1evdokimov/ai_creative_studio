from .base import LLMBackend


class MLXBackend(LLMBackend):

    def __init__(self, name):
        from mlx_lm import load, generate
        
        self.model, self.tokenizer = load(
            name
        )


    def generate(self, prompt: str) -> str:

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Return only JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]


        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        stop_tokens=[
            "<|endoftext|>",
            "Human:",
            "Assistant:"
        ]


        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=300,
            # temperature=0.2,
            # stop_tokens=stop_tokens
        )


        return response