from .base import LLMBackend


class MockLLM(LLMBackend):

    def generate(self, prompt: str) -> str:

        print("\nLLM PROMPT:")
        print(prompt)

        return """
[
 {
  "name": "Luxury Spa",
  "scene": "marble bathroom",
  "lighting": "soft morning light",
  "style": "premium"
 },
 {
  "name": "Natural Beauty",
  "scene": "botanical garden",
  "lighting": "sunlight",
  "style": "organic"
 }
]
"""