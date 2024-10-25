from packaging.version import parse as parse_version

from torch import device

from ..utils.import_utils import compare_transformers_version

CPU = device("cpu")
CUDA_0 = device("cuda:0")

SUPPORTED_MODELS = ["bloom", "gptj", "gpt2", "gpt_neox", "gpt_neox_moe", "opt", "moss", "phi-msft", "llama_moe"]
if compare_transformers_version("v4.28.0", op="ge"):
    SUPPORTED_MODELS.append("llama")

EXPERT_ID_PLACEHOLDER = "EXPERT_ID"

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS", "EXPERT_ID_PLACEHOLDER"]
