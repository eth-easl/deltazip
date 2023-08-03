import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # just placeholder..., we don't need it for base model
    compress_config = BaseCompressionConfig(
            bits = 4,
            group_size=128,
            sparsity=1,
            prunen=0,
            prunem=0,
            lossless='gdeflate',
        )
    with torch.inference_mode():
        base_model = AutoFMZipModelForCausalLM.from_pretrained(
            args.base_model,
            compress_config=compress_config, torch_dtype=torch.float16
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
        
        prompt = "Alan Turing is "
        # add delta to base model
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
        print(
            tokenizer.decode(output[0], skip_special_tokens=True))
        end = timer()
        logger.info(f"[FMZip] Total time: {end - start} seconds")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--delta", type=str, choices=['subtract', 'xor'], default='subtract')
    args = parser.parse_args()
    main(args)