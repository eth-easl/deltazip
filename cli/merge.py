import os
import torch
import argparse
from loguru import logger
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from transformers import AutoTokenizer
compress_config = BaseCompressionConfig(
    bits=4,
    group_size=128,
    sparsity=1,
    prunen=0,
    prunem=0,
    lossless="gdeflate",
    damp_percent=0.02,
)

def merge(args):
    print(args)
    with torch.inference_mode():
        base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
            args.base_model, compress_config=compress_config
        )
        base_model = base_model.half()
        logger.info("Loading target model")
        delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
            args.target_model, strict=True, device="cpu", unpack=True
        )
        delta_model = delta_model.half()
        compressed_modules = []
        for x in base_model.inside_layer_modules:
            compressed_modules.extend(x)
        if args.delta == "subtract":
            for name, param in base_model.model.named_parameters():
                delta_model.model.state_dict()[name].copy_(
                    param + delta_model.model.state_dict()[name]
                )

        # save model to output directory
        for name, param in delta_model.model.state_dict().items():
            param = param.contiguous()
        delta_model.model.save_pretrained(args.output_dir, safe_serialization=False, max_shard_size="10GB")
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        tokenizer.save_pretrained(args.output_dir)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--delta", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    merge(args)
