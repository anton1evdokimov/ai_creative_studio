from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> str:
        """Generate response from language model.

        Implementations should gracefully ignore kwargs they don't support.
        """
