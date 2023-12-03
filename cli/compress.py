import os
import json
import torch
import argparse
from transformers import AutoTokenizer
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig


def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=args.fast_tokenizer
    )
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
    torch.cuda.empty_cache()
    # now time to prepare inspect dataset
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)["text"] for line in fp.readlines()]
    if args.n_samples <= 0:
        examples = examples
    else:
        if args.shuffle_dataset:
            import random
            random.seed(42)
            random.shuffle(examples)
        examples = examples[: args.n_samples]
    examples = [tokenizer(x) for x in examples]
    if args.base_model != "" and args.delta != "":
        target_model.lossy_compress(
            examples,
            batch_size=1,
            base_model=base_model,
        )
    else:
        target_model.lossy_compress(
            examples,
            batch_size=1,
        )
    # write to folder
    os.makedirs(args.outdir, exist_ok=True)
    # for weights that are not compressed, we calculate delta afterward compression
    if args.base_model != "" and args.delta != "":
        compressed_modules = []
        for x in base_model.inside_layer_modules:
            compressed_modules.extend(x)
        for name, param in target_model.named_parameters():
            if "bias" in name or all(
                [modules not in name for modules in compressed_modules]
            ):
                target_model.state_dict()[name].copy_(
                    param - base_model.state_dict()[name]
                )
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
    parser.add_argument("--fast-tokenizer", action="store_true")
    parser.add_argument("--shuffle-dataset", action="store_true")
    args = parser.parse_args()
    main(args)
