import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")

ft_model = AutoModelForCausalLM.from_pretrained("/mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/llama-3b/task112_asset_simple_sentence_identification/global_step24")

compressed_model_path = "/mnt/scratch/xiayao/cache/experiments/fmzip/compressed_models/3b0.75s/open_llama_3b_v2/task112_asset_simple_sentence_identification/global_step24"

# diff distribute, check how many percent of zeros in the delta
## step 1: get delta
total_numel = 0
total_zero = 0
for name, param in base_model.named_parameters():
    delta = param - ft_model.state_dict()[name]
    zero_percent = 100 * (delta.numel() - delta.nonzero().size(0)) / delta.numel()
    total_numel += delta.numel()
    total_zero += delta.numel() - delta.nonzero().size(0)
    print(f"{name}: {zero_percent:.2f}% zeros out of {delta.numel()} params")

print(f"On average, {100 * total_zero / total_numel:.2f}% zeros out of {total_numel} params")