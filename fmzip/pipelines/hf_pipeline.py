import torch
import transformers
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
        batch_size: int = 1,
    ) -> None:
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
        self.batch_size = batch_size

    def generate(self, queries: List[Tuple], **kwargs):
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
                    model = self._load_target_model(model_name, BASE_DEVICE)
                    loading_end = timer()
                    inference_start = timer()
                    output = model.generate(**batch_inputs, **kwargs)
                    inference_end = timer()
                    output = self.tokenizer.batch_decode(output)
                    tokenize_time = tokenize_end - tokenize_start
                    loading_time = loading_end - loading_start
                    prepare_time = 0
                    inference_time = inference_end - inference_start
                    output = [
                        {
                            "data": o,
                            "model": model_name,
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

    def _load_target_model(self, model_name: str, device: str):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        model = model.to(torch.device(device))
        return model

    def _prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(BASE_DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(BASE_DEVICE)
        return batch
