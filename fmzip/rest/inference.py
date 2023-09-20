import torch
import transformers
from typing import List
from fmzip.pipelines import MixedPrecisionModel

class InferenceService():
    def __init__(self, provider: str, **kwargs) -> None:
        self.provider = provider
        if provider=='fmzip':
            self.mpm = MixedPrecisionModel(**kwargs)
        elif provider=='hf':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(kwargs['base_model'])
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError
    
    def generate(self, queries: List):
        queries = [(query.prompt, query.model) for query in queries]
        if self.provider == 'fmzip':
            return self.mpm.generate(queries)
        elif self.provider == 'hf':
            outputs = []
            for query in queries:
                model = transformers.AutoModelForCausalLM.from_pretrained(query[1])
                model = model.to(torch.device("cuda"))
                batch = self.tokenizer([inp[0] for inp in queries], return_tensors="pt", padding=True)
                batch["input_ids"] = batch["input_ids"].to(torch.device("cuda"))
                batch["attention_mask"] = batch["attention_mask"].to(torch.device("cuda"))
                output = model.generate(**batch)
                output = self.tokenizer.batch_decode(
                    output,
                    skip_special_tokens=True
                )
                outputs.append(output)
            return outputs
        else:
            raise NotImplementedError