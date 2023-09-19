# Mixed Precision Generation
from fmzip.pipelines import MixedPrecisionModel
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse
from transformers import AutoTokenizer
import torch

base_model_name = 'EleutherAI/pythia-125m-deduped'
delta = ".cache/compressed_models/p125m_gsd_133"

def load_delta_model():
    compress_config = BaseCompressionConfig(
        bits = 4,
        group_size=128,
        sparsity=1,
        prunen=0,
        prunem=0,
        lossless='gdeflate',
        damp_percent=0.02
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoFMZipModelForCausalLM.from_pretrained(
        base_model_name,
        compress_config=compress_config
    )
    base_model = base_model.to(torch.device('cuda'))
    delta_model = AutoFMZipModelForCausalLM.from_compressed(
        delta, 
        strict=False,
        device='cpu',
        unpack=True
    )
    delta_model = delta_model.half()
    delta_model = delta_model.to(torch.device('cuda'))
    delta_model = subtract_inverse(base_model, delta_model)
    return delta_model, tokenizer

def main():
    delta_model, tokenizer = load_delta_model()
    mpm = MixedPrecisionModel("EleutherAI/pythia-125m-deduped")
    mpm.load_delta(".cache/compressed_models/p125m_gsd_133")
    test_data = [
        ("Computer Science is about ", ".cache/compressed_models/p125m_gsd_133"), 
        ("What is the weather today?", ".cache/compressed_models/p125m_gsd_133"),
        ("Alan Turing is ", ".cache/compressed_models/p125m_gsd_133"), 
        ("QED is ", ".cache/compressed_models/p125m_gsd_133")
    ]
    
    delta_model_results = []
    for i in test_data:
        output = delta_model.generate(
                **tokenizer(i[0], return_tensors="pt").to(delta_model.device),
        )
        delta_model_results.append(tokenizer.decode(output[0], skip_special_tokens=True))
    print("delta model results")
    print(delta_model_results)
    print("Mixed Precision Model results")
    results = mpm.generate(test_data)
    print(results)

if __name__=="__main__":
    main()