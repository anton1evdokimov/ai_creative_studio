import platform

from .mlx_backend import MLXBackend


def create_llm():

    system = platform.system()


    if system == "Darwin":

        print("Using MLX backend")

        return MLXBackend(
            "mlx-community/Qwen2.5-3B-Instruct-4bit"
        )


    raise RuntimeError(
        "No supported LLM backend"
    )