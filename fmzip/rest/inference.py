import torch
import transformers
from typing import List
from fmzip.pipelines import MixedPrecisionModel
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import subtract_inverse

class InferenceService():
    def __init__(self, provider: str, **kwargs) -> None:
        self.provider = provider
        compress_config = BaseCompressionConfig(
            bits = 4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless='gdeflate',
            damp_percent=0.02
        )
        if provider == 'fmzip':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(kwargs['base_model'])
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            base_model = AutoFMZipModelForCausalLM.from_pretrained(
                kwargs['base_model'],
                compress_config=compress_config
            )
            self.base_model = base_model.to(torch.device('cuda'))
        elif provider=='fmzip-mpm':
            self.mpm = MixedPrecisionModel(**kwargs)
        elif provider=='hf':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(kwargs['base_model'])
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            raise NotImplementedError
    
    def _mpm_generate(self, queries: List):
        return self.mpm.generate(queries)

    def _hf_generated(self, queries: List):
        outputs = []
        for query in queries:
            model = transformers.AutoModelForCausalLM.from_pretrained(query[1])
            model = model.to(torch.device("cuda"))
            batch = self.tokenizer(query[0], return_tensors="pt", padding=True)
            batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
            batch["attention_mask"] = batch["attention_mask"].to(torch.device("cuda"))
            output = model.generate(**batch)
            output = self.tokenizer.batch_decode(
                output,
                skip_special_tokens=True
            )
            outputs.append(output[0])
        return outputs
    
    def _fmzip_generate(self, queries: List):
        outputs = []
        for query in queries:
            delta_model = AutoFMZipModelForCausalLM.from_compressed(
                query[1],
                strict=False,
                device='cpu',
                unpack=True
            )
            delta_model = delta_model.to(torch.device('cuda'))
            delta_model = delta_model.half()
            delta_model = subtract_inverse(self.base_model, delta_model)
            batch = self.tokenizer(query[0], return_tensors="pt", padding=True)
            batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
            batch["attention_mask"] = batch["attention_mask"].to(torch.device("cuda"))
            output = delta_model.generate(**batch)
            output = self.tokenizer.batch_decode(
                output,
                skip_special_tokens=True
            )
            outputs.append(output[0])
        return outputs

    def generate(self, queries: List):
        queries = [(query.prompt, query.model) for query in queries]
        if self.provider == 'fmzip-mpm':
            return self.mpm.generate(queries)
        elif self.provider == 'hf':
            return self._hf_generated(queries)
        elif self.provider == 'fmzip':
            return self._fmzip_generate(queries)
        else:
            raise NotImplementedError