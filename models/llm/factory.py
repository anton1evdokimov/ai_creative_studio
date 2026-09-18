import platform
import torch

from models.llm.cuda_backend import CUDABackend
from .mlx_backend import MLXBackend


def create_llm():

    system = platform.system()


    # Mac Apple Silicon
    if system == "Darwin":

        print("Using MLX backend")

        return MLXBackend(
            "mlx-community/Qwen2.5-3B-Instruct-4bit"
        )


    # NVIDIA GPU
    if torch.cuda.is_available():

        print("Using CUDA backend")

        return CUDABackend(
            "Qwen/Qwen2.5-3B-Instruct"
        )


    raise RuntimeError(
        "No supported backend found"
    )