import torch
from loguru import logger
from typing import List, Tuple
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.modeling.gpt_neox import parallelize_neox
from fmzip.modeling.llama import parallelize_llama

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class MixedPrecisionModel:
    def __init__(self, base_model: str, max_num_deltas=2, use_bfloat16=False, batch_size=0) -> None:
        compress_config = BaseCompressionConfig(
            bits=4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless="gdeflate",
            damp_percent=0.02,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, use_fast=True)
        # https://github.com/facebookresearch/llama/issues/380
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"
        self.batch_size = batch_size
        with torch.device("cuda"):
            self.base_model = AutoFMZipModelForCausalLM.from_pretrained(
                base_model,
                compress_config=compress_config,
                low_cpu_mem_usage=True
            )
        if use_bfloat16:
            self.base_model = self.base_model.bfloat16()
        else:
            self.base_model = self.base_model.half()
        self.base_model = self.base_model.to(torch.device("cuda"))
        logger.info("based model loaded")
        self.model_pool = {}
        self.key_list = []
        self.max_num_deltas = max_num_deltas
        parallelize_neox()
        parallelize_llama()

    def generate(self, queries: List[Tuple], **kwargs):
        batch = self.prepare_batch(
            queries,
            self.tokenizer,
        )
        outputs = []
        for batch_idx in range(0, len(queries), self.batch_size):
            deltas = [x[1] for x in queries[batch_idx:batch_idx+self.batch_size]]
            batch_inputs = {
                key: batch[key][batch_idx:batch_idx+self.batch_size]
                for key in batch
            }
            # if delta is not in the model pool, load them
            # but before that, check if the model pool is full
            print(len(self.model_pool))
            if len(self.model_pool) + len(deltas) >= self.max_num_deltas:
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
            logger.info(f"prepare finished. Takes {end-start:.2f} seconds")
            output = self.base_model.generate(**batch_inputs, **kwargs)

            # remove the delta modules
            for key in self.key_list:
                _, target, _ = _get_submodules(self.base_model, key)
                delattr(target, "delta")
            
            output = self.tokenizer.batch_decode(
                output,
                skip_special_tokens = True
            )
            outputs.extend(output)
        # reorganize the outputs to tuple, (delta_name, outputs)
        return outputs

    def load_delta(self, delta_model: str):
        logger.info("Loading target model")
        start = timer()
        self.model_pool[delta_model] = AutoFMZipModelForCausalLM.from_compressed(
            delta_model,
            device='cuda',
            unpack = False,
            low_cpu_mem_usage=True
        )
        end = timer()
        logger.info(f"Loading finished. Takes {end-start:.2f} seconds")
        if len(self.key_list) == 0:
            for key, _ in self.model_pool[delta_model].model.named_modules():
                self.key_list.append(key)

    def prepare_batch(self, inputs, tokenizer):
        """Tokenizes inputs and sets the batch_lora_ids for the model."""
        batch = tokenizer([inp[0] for inp in inputs], return_tensors="pt", padding=True)
        batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
        batch["attention_mask"] = batch["attention_mask"].to(torch.device("cuda"))
        return batch

    def expire_delta(self, delta_model: str, expire_level=0):
        # if expire_level = 1
        # move the model to cpu
        # if expire_level = 2
        # delete the model from memory
        pass