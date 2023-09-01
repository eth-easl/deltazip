import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer
from timeit import default_timer as timer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse

def main(args):
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # just placeholder..., we don't need it for base model
    # compress_config = BaseCompressionConfig(
    #         bits = 4,
    #         group_size=128,
    #         sparsity=1,
    #         prunen=0,
    #         prunem=0,
    #         lossless='gdeflate',
    #         damp_percent=0.02
    #     )
    with torch.inference_mode():
        start = timer()
        #base_model = AutoFMZipModelForCausalLM.from_pretrained(
        #    args.base_model,
        #    compress_config=compress_config, torch_dtype=torch.float16
        #)
        # base_model = base_model.to(torch.device('cuda'))
        end = timer()
        logger.info(f"Loading base model took {end - start:.2f}s")
        logger.info("Loading target model")
        start = timer()
        delta_model = AutoFMZipModelForCausalLM.from_compressed(
            args.target_model, 
            strict=False,
            device='cpu',
            unpack=True
        )
        # check how sparse each parameter is
        sparsity = {}
        for name, param in delta_model.named_parameters():
            sparsity[name] = (param == 0).sum().item() / param.numel()
        logger.info(f"Sparsity: {sparsity}")
        with open("sparsity.json", "w") as fp:
            import json
            json.dump(sparsity, fp)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="EleutherAI/pythia-2.8b-deduped")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--delta", type=str, choices=['subtract', 'xor'], default='subtract')
    args = parser.parse_args()
    main(args)