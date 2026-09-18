from abc import ABC, abstractmethod


class LLMBackend(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate response from language model
        """
        pass