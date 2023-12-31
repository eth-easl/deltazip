import os
import json
import torch
import argparse
from transformers import AutoTokenizer
from deltazip.utils.delta_utils import subtract, xor
from deltazip import AutoDeltaZipModelForCausalLM, AutoCompressionConfig


def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    compress_config = AutoCompressionConfig(
        tolerance=args.tolerance,
        bits=args.bits,
        sparsity=args.sparsities,
        prunen=args.prunen,
        prunem=args.prunem,
        lossless=args.lossless,
        damp_percent=args.perc_damp,
    )
    print("[info] compress config:", compress_config)
    target_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        args.target_model, compress_config=compress_config, torch_dtype=torch.float16
    )
    target_model.requires_grad_(False)
    if args.base_model != "":
        # import copy
        # target_model_copy = copy.deepcopy(target_model)
        print("[info] base model is defined, delta mode enabled")
        base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
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
    # now time to prepare inspect dataset
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)["text"] for line in fp.readlines()]
    if args.n_samples <= 0:
        examples = examples
    else:
        import random

        examples = random.sample(examples, args.n_samples)
    examples = [tokenizer(x) for x in examples]
    target_model.auto_lossy_compress(examples)
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
        "--bits",
        nargs="+",
        help="<Required> search space for bits",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--sparsities",
        nargs="+",
        help="<Required> search space for sparsities",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1024,
        help="How many data samples used for calibration, -1 means all.",
    )
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument(
        "--lossless", type=str, default="gdeflate", choices=["gdeflate"]
    )
    parser.add_argument(
        "--delta", type=str, choices=["subtract", "xor"], default="subtract"
    )
    parser.add_argument("--perc-damp", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default=".cache/compressed_models")
    args = parser.parse_args()
    main(args)
