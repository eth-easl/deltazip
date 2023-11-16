import torch
import transformers
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import subtract_inverse, subtract

base_model = "openlm-research/open_llama_3b_v2"
target_model = "/mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/llama-3b/task372_synthetic_palindrome_numbers/global_step105/"
delta_model = "/mnt/scratch/xiayao/cache/experiments/fmzip/compressed_models/4b0s/open_llama_3b_v2/task372_synthetic_palindrome_numbers/global_step105"

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
    base_model = transformers.AutoModel.from_pretrained(
        base_model
    )
    base_model = base_model.half()
    base_model = base_model.to(torch.device("cuda"))

    target_model = transformers.AutoModel.from_pretrained(
        target_model
    )
    target_model = target_model.half()
    target_model = target_model.to(torch.device("cuda"))

    diff_model = subtract(base_model, target_model)
    
    reconstructed_model = subtract_inverse(base_model, diff_model)
    for name, param in reconstructed_model.state_dict().items():
        if not torch.equal(param, target_model.state_dict()[name]):
            print(f"param {name} is not the same")
        else:
            print(f"param {name} is the same")