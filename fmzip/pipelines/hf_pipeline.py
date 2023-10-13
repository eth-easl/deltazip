import torch
from typing import List, Tuple
from transformers import AutoTokenizer
from fmzip.pipelines.utils import get_available_gpus, get_gpu_count

placement_strategies = ["tensor-parallel", "no-parallel"]

DEFAULT_CUDA_DEVICE = 1 if get_gpu_count() > 1 else 0
BASE_DEVICE = torch.device("cuda", DEFAULT_CUDA_DEVICE)

class HuggingFacePipeline:
    def __init__(self, base_model: str, max_num_models: int = get_gpu_count(), batch_size: int=1) -> None:
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        # avoid using eos_token as padding token
        # https://github.com/facebookresearch/llama/issues/380
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"
        # model pool maps model name to gpu indices
        self.model_pool = {}
    
    def generate(self, queries: List[Tuple], **kwargs):
        pass

    def _prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(BASE_DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(BASE_DEVICE)
        return batch