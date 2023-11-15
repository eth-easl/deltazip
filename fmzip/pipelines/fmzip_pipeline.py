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
from fmzip.nn_modules.batched_qlinear import WarmupBQLForward

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
        warmup_models: List[str] = [],
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
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.tokenizer.padding_side = "left"
        # avoid using eos_token as padding token
        # https://github.com/facebookresearch/llama/issues/380
        # the core assumption of fmzip is that the base model is always loaded, so let's load it when initialize
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
        self.model_pool = {}
        self.req_count = {}
        self.key_list = []
        if len(warmup_models) > 0:
            logger.info("Warming up model triton...")
            self._load_deltas(warmup_models)
            WarmupBQLForward([self.model_pool[delta] for delta in warmup_models])
            self.model_pool = {}
            self.req_count = {}
            self.key_list = []
        
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, queries: List[Tuple], gpu_id: int = 0, **kwargs):
        outputs = []
        for batch_idx in range(0, len(queries), self.batch_size):
            tokenize_start = timer()
            sub_queries = queries[batch_idx : batch_idx + self.batch_size]
            batch = self._prepare_batch(sub_queries, self.tokenizer)
            deltas = [x[1] for x in sub_queries]
            batch_inputs = {k: batch[k] for k in batch}
            tokenize_end = timer()
            for delta in deltas:
                if delta not in self.req_count and delta != self.base_model_name:
                    self.req_count[delta] = 0
                elif delta in self.req_count:
                    self.req_count[delta] += 1
            for k in batch_inputs:
                batch_inputs[k] = batch_inputs[k].to(f"cuda:{gpu_id}")
            loading_start = timer()
            # check if eviction needed, ensure enough memory for loading new deltas
            self._evict_deltas(deltas)
            self.report_meminfo()
            self._load_deltas(deltas, self.offload_base_model, int(gpu_id))
            loading_end = timer()
            prepare_start = timer()
            self._prepare_inference(deltas, int(gpu_id))
            prepare_end = timer()
            inference_start = timer()
            kwargs["do_sample"] = False
            torch.cuda.profiler.cudart().cudaProfilerStart()
            output = self.base_models[int(gpu_id)].generate(**batch_inputs, **kwargs)
            torch.cuda.profiler.cudart().cudaProfilerStop()
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
                self._clear_addback_delta(deltas[0], int(gpu_id))
            elif self.placement_strategy in ["colocate", "separation"]:
                self._clear_colocate(int(gpu_id))
            # torch.cuda.empty_cache()
        return outputs

    def _load_base_model(self):
        self.base_models = [None for _ in range(self.device_count)]
        logger.info("loading base model")
        for gpu_id in range(self.device_count):
            with torch.device("cuda", gpu_id):
                self.base_models[gpu_id] = AutoFMZipModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    compress_config=dummy_compression_config,
                    low_cpu_mem_usage=True,
                )
            self.base_models[gpu_id] = self.base_models[gpu_id].to(gpu_id)
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
            use_triton=True if not self.placement_strategy == "colocate" else False,
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
        return batch

    def _offload_delta(self, delta):
        self.model_pool[delta] = self.model_pool[delta].to("cpu")
        torch.cuda.empty_cache()

    def _load_deltas(self, deltas: List[str], offload_base_model=False, gpu_id=0):
        if all([delta == self.base_model_name for delta in deltas]):
            logger.info("loading skipped: all requested models are base model")
            return
        if offload_base_model:
            logger.info("offloading base model")
            self.base_models[gpu_id] = self.base_models[gpu_id].to(
                "cpu", non_blocking=True
            )
            logger.info("base model offloaded")
        if self.placement_strategy in ["colocate", "addback"]:
            [
                self._load_delta(delta, device=f"cuda:{gpu_id}")
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
            self.base_models[gpu_id] = self.base_models[gpu_id].to(
                gpu_id, non_blocking=True
            )

    def report_meminfo(self):
        # force garbage collection and free memory
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        free, total = torch.cuda.mem_get_info()
        logger.warning(
            f"allocated: pytorch/cupy {torch.cuda.memory_allocated() / 1e9:.2f}/{cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB"
        )
        logger.warning(f"PyTorch free/total: {free / 1e9:.2f}/{total / 1e9:.2f} GB")

    def _evict_deltas(self, deltas: List[str]):
        print(f"len model pool: {len(self.model_pool)}")
        if len(self.model_pool) + len(deltas) > self.max_num_deltas:
            req_count_loaded = {
                k: v for k, v in self.req_count.items() if k in self.model_pool
            }
            print(f"req_count_loaded: {req_count_loaded}")
            # sort req_count by value
            to_evict_models = sorted(req_count_loaded, key=req_count_loaded.get)[
                : len(deltas)
            ]
            logger.info(f"evicting {to_evict_models}")
            for delta in to_evict_models:
                try:
                    del self.model_pool[delta]
                except KeyError:
                    pass

    def _prepare_inference(self, deltas, gpu_id):
        if self.placement_strategy == "addback":
            self._prepare_addback(deltas, gpu_id)
        elif self.placement_strategy in ["colocate", "separation"]:
            self._prepare_colocate(deltas, gpu_id)
        else:
            raise ValueError(
                f"Unsupported placement strategy: {self.placement_strategy}"
            )

    def _prepare_addback(self, deltas: List[str], gpu_id: int = 0):
        # add the delta back to the base model, this only supports len(deltas)=1
        assert len(deltas) == 1, "addback only supports len(deltas)=1"

        if deltas[0] != self.base_model_name:
            logger.info(f"adding delta {deltas[0]} to base model")
            with torch.no_grad():
                for name, param in self.model_pool[deltas[0]].model.named_parameters():
                    self.base_models[gpu_id].state_dict()[name] += param

    def _prepare_colocate(self, deltas, gpu_id):
        if all([delta == self.base_model_name for delta in deltas]):
            for key, dmodule in self.base_models[gpu_id].named_modules():
                setattr(dmodule, "delta", [None for delta in deltas])

        for key in self.key_list:
            _, target, _ = get_submodules(self.base_models[gpu_id], key)
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

    def _clear_addback_delta(self, delta, gpu_id: int):
        if delta != self.base_model_name:
            logger.info(f"clearing delta {delta} from base model")
            # remove the delta part from the base_model again
            with torch.no_grad():
                for name, param in self.model_pool[delta].model.named_parameters():
                    self.base_models[gpu_id].state_dict()[name] -= param.to(gpu_id)

    def _clear_colocate(self, gpu_id):
        for key, dmodule in self.base_models[gpu_id].named_modules():
            setattr(dmodule, "delta", [])

    def find_model(self, model_name):
        if model_name in self.model_pool:
            return self.model_pool[model_name].device
        else:
            return None
