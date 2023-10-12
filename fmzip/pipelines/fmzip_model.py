import os
import torch
from loguru import logger
from typing import List, Tuple
from transformers import AutoTokenizer
from timeit import default_timer as timer

from fmzip.modeling.llama import parallelize_llama
from fmzip.modeling.gpt_neox import parallelize_neox
from fmzip.pipelines.utils import get_available_gpus, get_gpu_count, get_submodules
from fmzip import BaseCompressionConfig, AutoFMZipModelForCausalLM
from fmzip.utils.delta_utils import subtract_inverse

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
        # https://github.com/facebookresearch/llama/issues/380
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"
        # the core assumption of fmzip is that the base model is always loaded, so let's load it when initialize
        self._load_base_model()
        # fmzip manages a pool of model
        self.model_pool = {}
        self.key_list = []
        if self.placement_strategy != "addback":
            parallelize_neox()
            parallelize_llama()

    def _load_base_model(self):
        logger.info("loading base model")
        with torch.device("cuda", DEFAULT_CUDA_DEVICE):
            self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
                self.base_model,
                compress_config=dummy_compression_config,
                low_cpu_mem_usage=True,
            )
        logger.info("based model loaded")

    def _load_delta(self, delta_model: str, device="cuda"):
        logger.info(f"Loading target model {delta_model} to {device}")
        self.model_pool[delta_model] = AutoFMZipModelForCausalLM.from_compressed(
            delta_model,
            device=device,
            unpack=True if self.placement_strategy == "addback" else False,
            low_cpu_mem_usage=False,
        )

    def _prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(BASE_DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(BASE_DEVICE)
        return batch

    def generate(self, queries: List[Tuple], **kwargs):
        tokenize_start = timer()
        batch = self._prepare_batch(queries, self.tokenizer)
        tokenize_end = timer()
        outputs = []
        for batch_idx in range(0, len(queries), self.batch_size):
            deltas = [x[1] for x in queries[batch_idx : batch_idx + self.batch_size]]
            batch_inputs = {
                k: batch[k][batch_idx : batch_idx + self.batch_size] for k in batch
            }
            # check if eviction needed, ensure enough memory for loading new deltas
            self._evict_deltas(deltas)
            loading_start = timer()
            self._load_deltas(deltas)
            loading_end = timer()
            prepare_start = timer()
            self._prepare_inference()
            prepare_end = timer()

    def _offload_deltas(self):
        pass

    def _load_deltas(self, deltas: List[str]):
        if self.placement_strategy in ["colocate", "addback"]:
            logger.warning(f"I am forcing reloading of all deltas")
            [self._load_delta(delta, device=DEFAULT_CUDA_DEVICE) for delta in deltas]
        elif self.placement_strategy == "separation":
            target_device = [i for i in range(self.device_count)]
            for i, delta in enumerate(deltas):
                target_device = target_device[i % self.device_count]
                if True:
                    logger.info(f"loading delta to device cuda:{target_device}")
                    self._load_delta(delta, device=target_device)
        else:
            raise ValueError(
                f"Unsupported placement strategy: {self.placement_strategy}"
            )

    def _evict_deltas(self, deltas: List[str]):
        if len(self.model_pool) + len(deltas) >= self.max_num_deltas:
            logger.info(f"model pool is full, evict all deltas")
            logger.warning(f"todo: implement LRU cache")
            self.model_pool = {}

    def _prepare_inference(self, deltas):
        if self.placement_strategy == "addback":
            self._prepare_inference_addback(deltas)
        elif self.placement_strategy in ["colocate", "separation"]:
            self._prepare_colocate(deltas)
        else:
            raise ValueError(
                f"Unsupported placement strategy: {self.placement_strategy}"
            )

    def _prepare_inference_addback(self, deltas: List[str]):
        # add the delta back to the base model, this only supports len(deltas)=1
        assert len(deltas) == 1, "addback only supports len(deltas)=1"
        self.base_model = subtract_inverse(self.base_model, self.model_pool[deltas[0]])

    def _prepare_colocate(self, deltas):
        for key in self.key_list:
            _, target, _ = get_submodules(self.base_model, key)
            dmodules = []
            for delta in deltas:
                for dkey, dmodule in self.model_pool[delta].model.named_modules():
                    if dkey == key:
                        dmodules.append(dmodule)
                        break
            setattr(target, "delta", dmodules)