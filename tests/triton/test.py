import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    with torch.inference_mode():
        logger.info("Loading target model")
        
        delta_model = AutoFMZipModelForCausalLM.from_compressed(
            args.target_model, strict=False, device="cuda", unpack=False
        )
        delta_model = delta_model.half()
        delta_model = delta_model.to(torch.device("cuda"))
        logger.info("ready to generate")
        prompt = "Alan Turing is "
        # add delta to base model
        start = timer()
        output = delta_model.generate(
            **tokenizer(prompt, return_tensors="pt").to(delta_model.device),
            do_sample=True,
            top_p=0.9,
            top_k=0,
            temperature=0.1,
            max_length=100,
            min_length=10,
            num_return_sequences=1,
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        end = timer()
        logger.info(f"[FMZip] Total time: {end - start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    args = parser.parse_args()
    main(args)
