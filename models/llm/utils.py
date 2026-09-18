import platform
import torch

def get_backend():

    system = platform.system()


    if system == "Darwin":
        return "mlx"


    if torch.cuda.is_available():
        return "cuda"


    return "cpu"