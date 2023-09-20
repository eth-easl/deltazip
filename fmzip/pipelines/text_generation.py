import torch
from loguru import logger
from typing import List, Tuple
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.modeling.gpt_neox import parallelize_neox

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class MixedPrecisionModel:
    def __init__(self, base_model: str, max_num_deltas=10) -> None:
        compress_config = BaseCompressionConfig(
            bits=4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless="gdeflate",
            damp_percent=0.02,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
            base_model,
            compress_config=compress_config
        )
        self.base_model = self.base_model.to(torch.device("cuda"))
        logger.info("based model loaded")
        self.model_pool = {}
        self.key_list = []
        self.max_num_deltas = max_num_deltas

    def generate(self, queries: List[Tuple], **kwargs):
        parallelize_neox()
        batch = self.prepare_batch(
            queries,
            self.tokenizer,
        )
        deltas = [x[1] for x in queries]
        # if delta is not in the model pool, load them
        # but before that, check if the model pool is full

        if len(self.model_pool) >= self.max_num_deltas:
            logger.info("model pool is full, removing the previous model")
            logger.warning("in future this will be replaced with LRU cache")
            self.model_pool = {}

        [self.load_delta(delta) for delta in deltas if delta not in self.model_pool]
        start = timer()

        for key in self.key_list:
            _, target, _ = _get_submodules(self.base_model, key)
            dmodules = []
            for delta in deltas:
                for dkey, dmodule in self.model_pool[delta].model.named_modules():
                    if dkey == key:
                        dmodules.append(dmodule)
                        break
            setattr(
                target,
                "delta",
                [module.to(torch.device("cuda")) for module in dmodules],
            )
        end = timer()
        logger.info(f"prepare finished. Takes {end-start} seconds")
        output = self.base_model.generate(**batch, **kwargs)
        output = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )
        return output

    def load_delta(self, delta_model: str):
        logger.info("Loading target model")
        start = timer()
        self.model_pool[delta_model] = AutoFMZipModelForCausalLM.from_compressed(
            delta_model,
            unpack=False,
        )
        end = timer()
        logger.info(f"Loading finished. Takes {end-start} seconds")
        if len(self.key_list) == 0:
            for key, _ in self.model_pool[delta_model].model.named_modules():
                self.key_list.append(key)

    def prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
        batch["attention_mask"] = batch["attention_mask"].to(torch.device("cuda"))
        # inp_loras = [lora_map[inp[1]] for inp in inputs]
        # for _, module in model.named_modules():
        #     module.batch_lora_ids = inp_loras
        return batch