from packaging.version import parse as parse_version

from torch import device

CPU = device("cpu")
CUDA_0 = device("cuda:0")

SUPPORTED_MODELS = ["llama", "gemma", "gemma2"]

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS"]
