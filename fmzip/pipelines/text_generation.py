import torch
from loguru import logger
from typing import List, Dict
from transformers import AutoTokenizer
from timeit import default_timer as timer

from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class MixedPrecisionModel():
    def __init__(self, base_model: str) -> None:
        compress_config = BaseCompressionConfig(
            bits = 4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless='gdeflate',
            damp_percent=0.02
        )
        self.tokenizer =  AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
            base_model,
            compress_config=compress_config
        )
        self.base_model = self.base_model.to(torch.device('cuda'))
        logger.info("based model loaded")
        self.model_pool = {}

    def generate(self, queries: List[Dict]):
        key_list = [key for key, _ in self.base_model.named_modules()]
        for query in queries:
            delta = self.model_pool[query['model']]
            for key, _ in delta.named_modules():
                print(_get_submodules(delta, key))
                

    def load_delta(self, delta_model: str):
        logger.info("Loading target model")
        start = timer()
        self.model_pool[delta_model] = AutoFMZipModelForCausalLM.from_compressed(
            delta_model,
            unpack=False,
        )
        end = timer()
        logger.info(f"Loading finished. Takes {end-start} seconds")
    
    def forward(self, **kwargs):
        pass