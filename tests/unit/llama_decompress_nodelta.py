import torch
import transformers
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse, subtract

base_model = "openlm-research/open_llama_3b_v2"
target_model = "/mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/llama-3b/task372_synthetic_palindrome_numbers/global_step105/"
delta_model = "/mnt/scratch/xiayao/cache/experiments/fmzip/compressed_models/4b0s_nodelta/open_llama_3b_v2/task372_synthetic_palindrome_numbers/global_step105"

with torch.inference_mode():
    compress_config = BaseCompressionConfig(
        bits=4,
        group_size=128,
        sparsity=1,
        prunen=0,
        prunem=0,
        lossless="gdeflate",
        damp_percent=0.02,
    )

    target_model = AutoFMZipModelForCausalLM.from_pretrained(
        target_model, compress_config=compress_config
    )
    target_model = target_model.half()
    target_model = target_model.to(torch.device("cuda"))
    
    print(f"Loading delta model...")
    delta_model = AutoFMZipModelForCausalLM.from_compressed(
        delta_model, 
        strict=True, 
        device="cpu",
        unpack=True
    )
    delta_model = delta_model.half()
    delta_model = delta_model.to(torch.device("cuda"))

    for name, param in delta_model.state_dict().items():
        print(f"{name}, {torch.max(param - target_model.state_dict()[name])}")
