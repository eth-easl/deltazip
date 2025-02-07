from torch import device
from packaging.version import parse as parse_version


CPU = device("cpu")
CUDA_0 = device("cuda:0")

SUPPORTED_MODELS = ["llama", "gemma", "gemma2"] + ["bloom", "gptj", "gpt2", "gpt_neox", "gpt_neox_moe", "opt", "moss", "phi-msft", "llama_moe", "llama_btc"]

EXPERT_ID_PLACEHOLDER = "EXPERT_ID"

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS", "EXPERT_ID_PLACEHOLDER"]
