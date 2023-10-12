import torch
import transformers
from typing import List
from loguru import logger
from timeit import default_timer as timer
from fmzip.pipelines import MixedPrecisionModel
from fmzip.utils.delta_utils import subtract_inverse
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig

AVAILBLE_MODEL_PARALLEL_STRATEGY = ["none", "separate", "sharded", "fully-sharded"]


class InferenceService:
    def __init__(
        self,
        provider: str,
        model_parallel_strategy: str = "none",
        eviction_strategy: str = "none",
        **kwargs,
    ) -> None:
        self.provider = provider
        compress_config = BaseCompressionConfig(
            bits=4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless="gdeflate",
            damp_percent=0.02,
        )
        self.model_parallel_strategy = model_parallel_strategy
        if provider == "fmzip":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                kwargs["base_model"]
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            base_model = AutoFMZipModelForCausalLM.from_pretrained(
                kwargs["base_model"], compress_config=compress_config
            )
            self.base_model = base_model.to(torch.device("cuda"))
        elif provider == "fmzip-mpm":
            self.mpm = MixedPrecisionModel(
                model_parallel_strategy=self.model_parallel_strategy, **kwargs
            )
        elif provider == "hf":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                kwargs["base_model"]
            )
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.tokenizer.padding_side = "left"
        else:
            raise NotImplementedError

    def _mpm_generate(self, queries: List):
        return self.mpm.generate(queries)

    def _hf_generated(self, queries: List, **kwargs):
        outputs = []
        for query in queries:
            with torch.device("cuda"):
                loading_start = timer()
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    query[1], torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
                model = model.to(torch.device("cuda"))
                loading_end = timer()
                loading_time = loading_end - loading_start
                tokenize_start = timer()
                batch = self.tokenizer(query[0], return_tensors="pt", padding=True)
                batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
                batch["attention_mask"] = batch["attention_mask"].to(
                    torch.device("cuda")
                )
                tokenize_end = timer()
                tokenize_time = tokenize_end - tokenize_start
                inference_start = timer()
                output = model.generate(**batch, **kwargs)
                output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                inference_end = timer()
                inference_time = inference_end - inference_start
                outputs.append(
                    {
                        "data": output[0],
                        "measure": {
                            "tokenize_time": tokenize_time,
                            "loading_time": loading_time,
                            "prepare_time": 0,
                            "inference_time": inference_time,
                        },
                    }
                )
        return outputs

    def _fmzip_generate(self, queries: List, **kwargs):
        outputs = []
        for query in queries:
            delta_model = AutoFMZipModelForCausalLM.from_compressed(
                query[1], strict=False, device="cpu", unpack=True
            )
            delta_model = delta_model.to(torch.device("cuda"))
            delta_model = delta_model.half()
            delta_model = subtract_inverse(self.base_model, delta_model)
            batch = self.tokenizer(query[0], return_tensors="pt", padding=True)
            batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
            batch["attention_mask"] = batch["attention_mask"].to(torch.device("cuda"))
            output = delta_model.generate(**batch, **kwargs)
            output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            outputs.append(output[0])
            logger.info("generation ends")
        return outputs

    def generate(self, queries: List, max_new_tokens=256):
        queries = [(query.prompt, query.model) for query in queries]
        if self.provider == "fmzip-mpm":
            return self.mpm.generate(
                queries, max_new_tokens=max_new_tokens, min_length=max_new_tokens
            )
        elif self.provider == "hf":
            return self._hf_generated(
                queries, max_new_tokens=max_new_tokens, min_length=max_new_tokens
            )
        elif self.provider == "fmzip":
            return self._fmzip_generate(
                queries, max_new_tokens=max_new_tokens, min_length=max_new_tokens
            )
        else:
            raise NotImplementedError
