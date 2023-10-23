import torch
import transformers
from loguru import logger
from typing import List, Tuple
from fmzip.pipelines.utils import get_available_gpus, get_gpu_count
from timeit import default_timer as timer

placement_strategies = ["tensor-parallel", "no-parallel"]

DEFAULT_CUDA_DEVICE = 1 if get_gpu_count() > 1 else 0
BASE_DEVICE = torch.device("cuda", DEFAULT_CUDA_DEVICE)

class HuggingFacePipeline:
    def __init__(
        self,
        base_model: str,
        max_num_models: int = get_gpu_count(),
        base_model_placement_strategy: str = "replication",
        batch_size: int = 1,
    ) -> None:
        self.current_model = None
        self.loaded_models = {}
        self.loaded_model_names = set()
        self.base_model = base_model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model, use_fast=True
        )
        # avoid using eos_token as padding token
        # https://github.com/facebookresearch/llama/issues/380
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"
        # model pool maps model name to gpu indices
        self.model_pool = {}
        self.device_models = {k: [] for k in range(get_gpu_count())}
        self.batch_size = batch_size

    def generate(self, queries: List[Tuple], gpu_id:int=0, **kwargs):
        with torch.inference_mode():
            tokenize_start = timer()
            batch = self._prepare_batch(queries, self.tokenizer)
            tokenize_end = timer()
            outputs = []
            for batch_idx in range(0, len(queries), self.batch_size):
                model_names = [
                    x[1] for x in queries[batch_idx : batch_idx + self.batch_size]
                ]
                batch_inputs = {
                    k: batch[k][batch_idx : batch_idx + self.batch_size] for k in batch
                }
                # construct inference pipeline
                loading_start = timer()
                for model_name in model_names:
                    model_device = self._load_target_model(model_name, gpu_id)
                    # move batch to device
                    for k in batch_inputs:
                        batch_inputs[k] = batch_inputs[k].to(f"cuda:{model_device}")
                    loading_end = timer()
                    inference_start = timer()
                    output = self.loaded_models[model_name].generate(**batch_inputs, **kwargs)
                    inference_end = timer()
                    output = self.tokenizer.batch_decode(output)
                    tokenize_time = tokenize_end - tokenize_start
                    loading_time = loading_end - loading_start
                    prepare_time = 0
                    inference_time = inference_end - inference_start
                    total_time = inference_end - tokenize_start
                    output = [
                        {
                            "data": o,
                            "model": model_name,
                            "measure": {
                                "tokenize_time": tokenize_time,
                                "loading_time": loading_time,
                                "prepare_time": prepare_time,
                                "inference_time": inference_time,
                                "total_time": total_time,
                            },
                        }
                        for o in output
                    ]
                    outputs.extend(output)
            return outputs

    def _find_device(self):
        # find device with minimal models
        # self.device_models maps from <device idx> -> [model names]
        # sort by number of models
        sorted_device_models = sorted(
            self.device_models.items(), key=lambda x: len(x[1])
        )
        return sorted_device_models[0][0]

    def _load_target_model(self, model_name: str, gpu_id=None):
        logger.info(f"loading {model_name}...")
        logger.info(f"loaded models: {self.loaded_model_names}")
        if model_name not in self.loaded_model_names:
            if gpu_id is None:
                model_device = self._find_device()
            else:
                model_device = gpu_id
            with torch.device(model_device):
                self.loaded_models[model_name] = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
                self.loaded_models[model_name] = self.loaded_models[model_name].to(torch.device(f"cuda:{model_device}"))
                self.device_models[model_device].append(model_name)
                self.loaded_model_names.add(model_name)
        else:
            logger.info(f"{model_name} already loaded, skipping...")
            model_device = None
            for device, models in self.device_models.items():
                if model_name in models:
                    model_device = device
                    break
        return model_device

    def _prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        return batch

    def find_model(self, model_name):
        for device, models in self.device_models.items():
            if model_name in models:
                return device
        return None