import os
import json
import argparse
from typing import Union
from transformers import AutoTokenizer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    delta_model = AutoFMZipModelForCausalLM.from_compressed(args.target_model, strict=False,device='cuda', unpack=True)
    

    prompt = "The meaning of life is"
    # add delta to base model
    base_model = AutoFMZipModelForCausalLM.from_pretrained(args.base_model, compress_config=delta_model.compress_config)
    base_model = base_model.to(delta_model.device)
    base_model.requires_grad_(False)

    if args.delta == "subtract":
        delta_model = subtract_inverse(base_model, delta_model.model)
    elif args.delta == "xor":
        delta_model = xor_inverse(base_model, delta_model.model)
    delta_model = delta_model.to('cuda')
    delta_model = delta_model.half()
    
    output = delta_model.generate(
        **tokenizer(prompt, return_tensors="pt").to(delta_model.device), 
        do_sample=True, 
        top_p=0.9, 
        top_k=0, 
        temperature=0.1, 
        max_length=100, 
        min_length=10, 
        num_return_sequences=1
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--delta", type=str, choices=['subtract', 'xor'], default='subtract')
    args = parser.parse_args()
    main(args)