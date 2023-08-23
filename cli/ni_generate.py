import os
import json
import torch
import argparse
from loguru import logger
from timeit import default_timer as timer
from transformers import AutoTokenizer, AutoModelForCausalLM
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse

def generate(args):
    print(args)
    # just placeholder, we don't need it for base model...
    # (todo:xiaozhe) remove the need of compress_config
    compress_config = BaseCompressionConfig(
        bits = 4,
        group_size=128,
        sparsity=1,
        prunen=0,
        prunem=0,
        lossless='gdeflate',
        damp_percent=0.02
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        base_model = AutoFMZipModelForCausalLM.from_pretrained(
            args.base_model,
            compress_config=compress_config
        )
        base_model = base_model.to(torch.device('cuda'))
        logger.info("Loading target model")
        start = timer()
        delta_model = AutoFMZipModelForCausalLM.from_compressed(
            args.target_model, 
            strict=False,
            device='cpu',
            unpack=True
        )
        delta_model = delta_model.half()
        delta_model = delta_model.to(torch.device('cuda'))
        logger.info("reverse delta")
        if args.delta == "subtract":
            delta_model = subtract_inverse(base_model, delta_model)
        elif args.delta == "xor":
            delta_model = xor_inverse(base_model, delta_model)
        logger.info("ready to generate")
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f]
        for datum in data:
            prompt = datum[args.input_field]
            # add delta to base model
            output = delta_model.generate(
                **tokenizer(prompt, return_tensors="pt").to(delta_model.device), 
                do_sample=args.do_sample, 
                max_new_tokens=args.max_length, 
                min_length=10, 
                num_return_sequences=1
            )
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            # remove the prompt from the output
            output = output.replace(prompt, "")
            datum["prediction"] = [output]
            
        with open(args.output_file, "w") as f:
            for datum in data:
                f.write(json.dumps(datum) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--delta", type=str, default="subtract")
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--input-field", type=str, default="input")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    generate(args)