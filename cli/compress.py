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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=args.fast_tokenizer)

    compress_config = BaseCompressionConfig(
        bits=args.bits,
        sparsity=args.sparsity,
        prunen=args.prunen,
        block_size=args.block_size,
        prunem=args.prunem,
        lossless=args.lossless,
        damp_percent=args.perc_damp,
        sym=False,
    )
    print("[info] compress config:", compress_config)
    target_model = AutoFMZipModelForCausalLM.from_pretrained(
        args.target_model, compress_config=compress_config, torch_dtype=torch.float16
    )
    target_model.requires_grad_(False)
    if args.base_model != "" and args.delta != "":
        print("[info] base model is defined, delta mode enabled")
        base_model = AutoFMZipModelForCausalLM.from_pretrained(
            args.base_model, compress_config=compress_config, torch_dtype=torch.float16
        )
        base_model.requires_grad_(False)
        base_model = base_model.to(torch.device("cuda"))
    torch.cuda.empty_cache()
    # now time to prepare inspect dataset
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)["text"] for line in fp.readlines()]
    if args.n_samples <= 0:
        examples = examples
    else:
        examples = examples[: args.n_samples]
    examples = [tokenizer(x) for x in examples]
    if args.base_model != "" and args.delta != "":
        target_model.lossy_compress(
            examples,
            batch_size=2,
            base_model=base_model,
        )
    else:
        target_model.lossy_compress(
            examples,
            batch_size=4,
        )
    # write to folder
    os.makedirs(args.outdir, exist_ok=True)
    target_model.save_compressed(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        default="answer_verification",
        help="The dataset to use for training, must be a path to a jsonl file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="How many data samples used for calibration, -1 means all.",
    )
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument(
        "--lossless", type=str, default="gdeflate", choices=["gdeflate"]
    )
    parser.add_argument("--delta", type=str, choices=["subtract", "xor"], default="")
    parser.add_argument("--perc-damp", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default=".cache/compressed_models")
    parser.add_argument("--fast-tokenizer", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
