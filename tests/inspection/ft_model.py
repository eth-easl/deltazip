import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024

finetune_task = "task112_asset_simple_sentence_identification"
base_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")

ft_model_base_path = f"/mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/llama-3b/{finetune_task}/"
compressed_model_path = f"/mnt/scratch/xiayao/cache/experiments/fmzip/compressed_models/3b0.75s/open_llama_3b_v2/{finetune_task}/"

steps = os.listdir(ft_model_base_path)
print(f"step,zero_percent,compression_ratio")
for step in steps:
    ft_model_path = os.path.join(ft_model_base_path, step)
    compressed_step_path = os.path.join(compressed_model_path, step)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(ft_model_path)
    # diff distribute, check how many percent of zeros in the delta
    ## step 1: get delta
    total_numel = 0
    total_zero = 0
    for name, param in base_model.named_parameters():
        delta = param - fine_tuned_model.state_dict()[name]
        zero_percent = 100 * (delta.numel() - delta.nonzero().size(0)) / delta.numel()
        total_numel += delta.numel()
        total_zero += delta.numel() - delta.nonzero().size(0)
    print(f"{step},{100 * total_zero / total_numel:.2f},{get_size(ft_model_path)/get_size(compressed_step_path):.2f}")
