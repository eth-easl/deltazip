import torch
import cupy as cp
from loguru import logger
from typing import List, Tuple
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip.modeling.llama import parallelize_llama
from fmzip.modeling.gpt_neox import parallelize_neox
from fmzip.pipelines.utils import get_gpu_count, get_submodules
from fmzip import BaseCompressionConfig, AutoFMZipModelForCausalLM

DEFAULT_CUDA_DEVICE = 1 if get_gpu_count() > 1 else 0
BASE_DEVICE = torch.device("cuda", DEFAULT_CUDA_DEVICE)

placement_strategies = ["addback", "colocate", "separation"]

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
        lossless_only: bool = False,
        offload_base_model: bool = False,
    ) -> None:
        if placement_strategy not in placement_strategies:
            raise ValueError(
                f"Unsupported placement strategy: {placement_strategy}, supported strategies are {placement_strategies}"
            )
        self.base_model_name = base_model
        self.offload_base_model = offload_base_model
        self.max_num_deltas = max_num_deltas
        self.batch_size = batch_size
        self.placement_strategy = placement_strategy
        self.placement_args = placement_args
        self.model_pool = {}
        self.req_count = {}
        self.key_list = []

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.tokenizer.padding_side = "left"
        # avoid using eos_token as padding token
        # https://github.com/facebookresearch/llama/issues/380
        # the core assumption of fmzip is that the base model is always loaded, so let's load it when initialize
        # fmzip manages a pool of model
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id

        self.device_count = get_gpu_count()
        self._load_base_model()
        self.lossless_only = lossless_only
        if self.lossless_only and self.placement_strategy != "addback":
            raise ValueError(
                f"lossless_only is only supported when placement_strategy is addback"
            )
        if self.placement_strategy != "addback":
            parallelize_neox()
            parallelize_llama()

    def generate(self, queries: List[Tuple], **kwargs):
        with torch.inference_mode():
            outputs = []
            for batch_idx in range(0, len(queries), self.batch_size):
                tokenize_start = timer()
                sub_queries = queries[batch_idx : batch_idx + self.batch_size]
                batch = self._prepare_batch(sub_queries, self.tokenizer)
                deltas = [
                    x[1] for x in sub_queries
                ]
                batch_inputs = {
                    k: batch[k] for k in batch
                }
                tokenize_end = timer()
                for delta in deltas:
                    if delta not in self.req_count and delta != self.base_model_name:
                        self.req_count[delta] = 0
                    elif delta in self.req_count:
                        self.req_count[delta] += 1
                loading_start = timer()
                # check if eviction needed, ensure enough memory for loading new deltas
                self._evict_deltas(deltas)
                self.report_meminfo()
                self._load_deltas(deltas, self.offload_base_model)
                loading_end = timer()
                prepare_start = timer()
                self._prepare_inference(deltas)
                prepare_end = timer()
                inference_start = timer()
                kwargs['do_sample'] = False
                output = self.base_model.generate(**batch_inputs, **kwargs)
                inference_end = timer()
                output = self.tokenizer.batch_decode(output)
                tokenize_time = tokenize_end - tokenize_start
                loading_time = loading_end - loading_start
                prepare_time = prepare_end - prepare_start
                inference_time = inference_end - inference_start
                total_time = inference_end - tokenize_start
                output = [
                    {
                        "data": o,
                        "model": deltas[i],
                        "measure": {
                            "tokenize_time": tokenize_time,
                            "loading_time": loading_time,
                            "prepare_time": prepare_time,
                            "inference_time": inference_time,
                            "total_time": total_time,
                        },
                    }
                    for i, o in enumerate(output)
                ]
                outputs.extend(output)
                if self.placement_strategy == "addback":
                    # for add back: batch size always 1
                    self._clear_addback_delta(deltas[0])
                elif self.placement_strategy in ["colocate", "separation"]:
                    self._clear_colocate()
                torch.cuda.empty_cache()
            return outputs

    def _load_base_model(self):
        logger.info("loading base model")
        with torch.device("cuda", DEFAULT_CUDA_DEVICE):
            self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
                self.base_model_name,
                compress_config=dummy_compression_config,
                low_cpu_mem_usage=True,
            )
        self.base_model = self.base_model.to(BASE_DEVICE)
        logger.info("based model loaded")

    def _load_delta(self, delta_model: str, device="cuda", force=False):
        if delta_model in self.model_pool and not force:
            logger.info(f"delta model {delta_model} already loaded")
            return
        logger.info(f"Loading target model {delta_model} to cuda:{device}")
        self.model_pool[delta_model] = AutoFMZipModelForCausalLM.from_compressed(
            delta_model,
            device=device,
            unpack=True
            if self.placement_strategy == "addback" and not self.lossless_only
            else False,
            low_cpu_mem_usage=True,
        )
        if delta_model not in self.req_count:
            self.req_count[delta_model] = 0
        if self.placement_strategy == "addback":
            self.model_pool[delta_model] = self.model_pool[delta_model].half()
        else:
            if len(self.key_list) == 0:
                # we need to figure out what to merge at this stage
                for key, _ in self.model_pool[delta_model].model.named_modules():
                    self.key_list.append(key)

    def _prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(BASE_DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(BASE_DEVICE)
        return batch

    def _offload_delta(self, delta):
        self.model_pool[delta] = self.model_pool[delta].to("cpu")
        torch.cuda.empty_cache()

    def _load_deltas(self, deltas: List[str], offload_base_model=False):
        if all([delta == self.base_model_name for delta in deltas]):
            logger.info("loading skipped: all requested models are base model")
            return
        if offload_base_model:
            self.base_model = self.base_model.to("cpu")
        if self.placement_strategy in ["colocate", "addback"]:
            [
                self._load_delta(delta, device=DEFAULT_CUDA_DEVICE)
                for delta in deltas
                if delta not in self.model_pool and delta != self.base_model_name
            ]
        elif self.placement_strategy == "separation":
            target_device = [i for i in range(self.device_count)]
            for i, delta in enumerate(deltas):
                target_device = target_device[i % self.device_count]
                logger.info(f"loading delta to device cuda:{target_device}")
                self._load_delta(delta, device=target_device)
        else:
            raise ValueError(
                f"Unsupported placement strategy: {self.placement_strategy}"
            )
        if offload_base_model:
            # move back
            self.base_model = self.base_model.to(BASE_DEVICE)

    def report_meminfo(self):
        # force garbage collection and free memory
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        free, total = torch.cuda.mem_get_info()
        logger.warning(
            f"PyTorch allocated memory: {torch.cuda.memory_allocated() / 1e9} GB"
        )
        logger.warning(
            f"Cupy allocated memory: {cp.get_default_memory_pool().used_bytes() / 1e9} GB"
        )
        logger.warning(f"PyTorch free memory: {free / 1e9} GB")
        logger.warning(f"PyTorch total memory: {total / 1e9} GB")
    
    def _evict_deltas(self, deltas: List[str]):
        # if all deltas are base model, no need to evict
        if all([delta == self.base_model_name for delta in deltas]):
            logger.info("eviction skipped: all requested models are base model")
            return
        if len(self.model_pool) + len(deltas) > self.max_num_deltas:
            logger.warning(f"Evicting {len(deltas)} models/deltas")
            # sort req_count by value
            to_evict_models = sorted(self.req_count, key=self.req_count.get)[
                : len(deltas)
            ]
            for delta in to_evict_models:
                del self.model_pool[delta]
                del self.req_count[delta]

    def _prepare_inference(self, deltas):
        if self.placement_strategy == "addback":
            self._prepare_addback(deltas)
        elif self.placement_strategy in ["colocate", "separation"]:
            self._prepare_colocate(deltas)
        else:
            raise ValueError(
                f"Unsupported placement strategy: {self.placement_strategy}"
            )

    def _prepare_addback(self, deltas: List[str]):
        # add the delta back to the base model, this only supports len(deltas)=1
        assert len(deltas) == 1, "addback only supports len(deltas)=1"
        
        if deltas[0] != self.base_model_name:
            logger.info(f"adding delta {deltas[0]} to base model")
            with torch.no_grad():
                for name, param in self.model_pool[deltas[0]].model.named_parameters():
                    self.base_model.state_dict()[name] += param

    def _prepare_colocate(self, deltas):
        if all([delta == self.base_model_name for delta in deltas]):
            for key, dmodule in self.base_model.named_modules():
                setattr(dmodule, "delta", [None for delta in deltas])

        for key in self.key_list:
            _, target, _ = get_submodules(self.base_model, key)
            dmodules = []
            for delta in deltas:
                if delta == self.base_model_name:
                    dmodules.append(None)
                else:
                    for dkey, dmodule in self.model_pool[delta].model.named_modules():
                        if dkey == key:
                            dmodules.append(dmodule)
                            break
            setattr(target, "delta", dmodules)

    def _clear_addback_delta(self, delta):
        if delta != self.base_model_name:
            logger.info(f"clearing delta {delta} from base model")
            # remove the delta part from the base_model again
            with torch.no_grad():
                for name, param in self.model_pool[delta].model.named_parameters():
                    self.base_model.state_dict()[name] -= param.to(BASE_DEVICE)

    def _clear_colocate(self):
        for key, dmodule in self.base_model.named_modules():
            setattr(dmodule, "delta", [])