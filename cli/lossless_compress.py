import os
import json
import torch
import argparse
from typing import Union
from transformers import AutoTokenizer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import subtract, xor


def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    compress_config = BaseCompressionConfig(
        bits=16,
        group_size=128,
        sparsity=0,
        prunen=0,
        prunem=0,
        lossless=args.lossless,
        damp_percent=0.01,
    )
    print("[info] compress config:", compress_config)
    target_model = AutoFMZipModelForCausalLM.from_pretrained(
        args.target_model, compress_config=compress_config, torch_dtype=torch.float16
    )
    target_model.requires_grad_(False)
    if args.base_model != "":
        # import copy
        # target_model_copy = copy.deepcopy(target_model)
        print("[info] base model is defined, delta mode enabled")
        base_model = AutoFMZipModelForCausalLM.from_pretrained(
            args.base_model, compress_config=compress_config
        )
        base_model.requires_grad_(False)

        # now perform the delta op
        if args.delta == "subtract":
            target_model = subtract(base_model, target_model)
        elif args.delta == "xor":
            target_model = xor(base_model, target_model)
        else:
            raise ValueError(f"Unknown delta mode: {args.delta}")
    for name, param in target_model.named_parameters():
        # check if nan exists
        if torch.isnan(param).any():
            raise ValueError(f"NaN exists in {name}")
    examples = []
    target_model.lossless_compress(examples)
    # write to folder
    os.makedirs(args.outdir, exist_ok=True)
    target_model.save_compressed(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument(
        "--lossless", type=str, default="gdeflate", choices=["gdeflate"]
    )
    parser.add_argument(
        "--delta", type=str, default="subtract", choices=["subtract", "xor"]
    )
    parser.add_argument("--outdir", type=str, default=".cache/compressed_models")
    args = parser.parse_args()
    main(args)
