import torch
from loguru import logger
from typing import List, Tuple
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip.pipelines.utils import get_gpu_count, get_available_gpus
from fmzip.modeling.llama import parallelize_llama
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.modeling.gpt_neox import parallelize_neox

DEFAULT_CUDA_DEVICE = 1 if get_gpu_count() > 1 else 0

BASE_DEVICE = torch.device("cuda", DEFAULT_CUDA_DEVICE)

# todo:xiaozhe this is only for llama model for now
inside_layer_modules = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class MixedPrecisionModel:
    def __init__(
        self,
        base_model: str,
        max_num_deltas=10,
        use_bfloat16=False,
        batch_size=0,
        model_parallel_strategy="none",
    ) -> None:
        compress_config = BaseCompressionConfig(
            bits=4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless="gdeflate",
            damp_percent=0.02,
        )
        self.device_count = len(get_available_gpus())
        logger.info(f"[fmzip] device count: {self.device_count}")
        self.model_parallel_strategy = model_parallel_strategy
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        # https://github.com/facebookresearch/llama/issues/380
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"
        self.batch_size = batch_size
        self.model_parallel_strategy = model_parallel_strategy
        with torch.device("cuda", DEFAULT_CUDA_DEVICE):
            self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
                base_model, compress_config=compress_config, low_cpu_mem_usage=True
            )
        if use_bfloat16:
            self.base_model = self.base_model.bfloat16()
        else:
            self.base_model = self.base_model.half()
        self.base_model = self.base_model.to(BASE_DEVICE)
        logger.info("based model loaded")
        self.model_pool = {}
        self.key_list = []
        self.max_num_deltas = max_num_deltas
        parallelize_neox()
        parallelize_llama()

    def generate(self, queries: List[Tuple], **kwargs):
        tokenize_start = timer()
        batch = self.prepare_batch(
            queries,
            self.tokenizer,
        )
        tokenize_end = timer()
        tokenize_time = tokenize_end - tokenize_start
        outputs = []
        for batch_idx in range(0, len(queries), self.batch_size):
            deltas = [x[1] for x in queries[batch_idx : batch_idx + self.batch_size]]
            batch_inputs = {
                key: batch[key][batch_idx : batch_idx + self.batch_size]
                for key in batch
            }
            # eviction
            if len(self.model_pool) + len(deltas) >= self.max_num_deltas:
                logger.info("model pool is full, removing the previous model")
                logger.warning("in future this will be replaced with LRU cache")
                self.model_pool = {}
            loading_start = timer()
            self.load_deltas(deltas)
            loading_end = timer()

            loading_time = loading_end - loading_start
            prepare_start = timer()
            for key in self.key_list:
                _, target, _ = _get_submodules(self.base_model, key)
                dmodules = []
                for delta in deltas:
                    # todo: fix this
                    for dkey, dmodule in self.model_pool[delta].model.named_modules():
                        if dkey == key:
                            dmodules.append(dmodule)
                            break
                setattr(
                    target,
                    "delta",
                    dmodules,
                )
            prepare_end = timer()
            prepare_time = prepare_end - prepare_start
            inference_start = timer()
            output = self.base_model.generate(**batch_inputs, **kwargs)
            inference_end = timer()
            inference_time = inference_end - inference_start
            for key in self.key_list:
                _, target, _ = _get_submodules(self.base_model, key)
                delattr(target, "delta")
            output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            output = [
                {
                    "data": o,
                    "measure": {
                        "tokenize_time": tokenize_time,
                        "loading_time": loading_time,
                        "prepare_time": prepare_time,
                        "inference_time": inference_time,
                    },
                }
                for o in output
            ]
            outputs.extend(output)
        return outputs

    def _load_delta(self, delta_model: str, device: str = "cuda"):
        logger.info(f"Loading target model {delta_model} to {device}")
        self.model_pool[delta_model] = AutoFMZipModelForCausalLM.from_compressed(
            delta_model, device=device, unpack=False, low_cpu_mem_usage=False
        )
        if len(self.key_list) == 0:
            # flatten insdier modules
            insider_modules = []
            [insider_modules.extend(x) for x in inside_layer_modules]
            # we need to figure out what to merge at this stage
            for key, _ in self.model_pool[delta_model].model.named_modules():
                self.key_list.append(key)

    def load_deltas(self, deltas: List[str]):
        if self.model_parallel_strategy == "separation":
            target_devices = [i for i in range(self.device_count)]
            for i, delta in enumerate(deltas):
                target_device = target_devices[i % len(target_devices)]
                #if delta not in self.model_pool:
                if True:
                    logger.info(f"loading delta to cuda:{target_device}")
                    self._load_delta(delta, device=f"cuda:{target_device}")

        elif self.model_parallel_strategy == "none":
            [
                self._load_delta(delta, device=f"cuda:{DEFAULT_CUDA_DEVICE}")
                for delta in deltas
                # (todo:xiaozhe): now enforce reloading
                if True
                #if delta not in self.model_pool
            ]
        else:
            raise ValueError(
                f"Unsupported model parallel strategy: {self.model_parallel_strategy}"
            )

    def prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(BASE_DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(BASE_DEVICE)
        return batch
