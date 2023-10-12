import os
import torch
from transformers import AutoTokenizer

from fmzip import BaseCompressionConfig, AutoFMZipModelForCausalLM
from fmzip.pipelines.utils import get_available_gpus, get_gpu_count
from loguru import logger

placement_strategy = ["addback", "colocate", "separation"]

DEFAULT_CUDA_DEVICE = 1 if get_gpu_count() > 1 else 0

BASE_DEVICE = torch.device("cuda", DEFAULT_CUDA_DEVICE)

dummy_compression_config = BaseCompressionConfig(
    bits=4,
    group_size=128,
    sparsity=1,
    prunen=0,
    prunem=0,
    lossless="gdeflate",
    damp_percent=0.02,
)


class FMZipPipeline:
    def __init__(
        self,
        base_model: str,
        max_num_deltas: int = 1,
        batch_size: int = 1,
        placement_strategy: str = "addback",
        placement_args: dict = None,
    ) -> None:
        self.base_model = base_model
        self.device_count = len(get_available_gpus())
        self.max_num_deltas = max_num_deltas
        self.batch_size = batch_size
        self.placement_strategy = placement_strategy
        self.placement_args = placement_args
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        # avoid using eos_token as padding token
        # # https://github.com/facebookresearch/llama/issues/380
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"
        # the core assumption of fmzip is that the base model is always loaded, so let's load it when initialize
        self._load_base_model()
        # fmzip manages a pool of model
        self.model_pool = {}
        self.key_list = []

    def _load_base_model(self):
        logger.info("loading base model")
        with torch.device("cuda", DEFAULT_CUDA_DEVICE):
            self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
                self.base_model,
                compress_config=dummy_compression_config,
                low_cpu_mem_usage=True,
            )
        logger.info("based model loaded")
