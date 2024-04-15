import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig, base_generation_strategies
from deltazip.modeling._const import EXPERT_ID_PLACEHOLDER
from loguru import logger


def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, use_fast=args.fast_tokenizer
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
    target_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        args.target_model, compress_config=compress_config, torch_dtype=torch.float16
    )
    target_model.requires_grad_(False)
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
    examples = [tokenizer(x, truncation=True, max_length=2048) for x in examples]
    examples = [e for e in examples if len(e['attention_mask']) != 0]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/base", exist_ok=True)
    
    logger.info("Saving base weights:")
    base_weights = target_model.get_moe_base_weights(base_generation_strategies.take_first)
    torch.save(base_weights, f"{args.outdir}/base/base_weights.pt")
    logger.info("Saving base weights finished")

    target_model.lossy_compress(
        examples,
        batch_size=1,
        is_moe=True
    )
    # write to folder
    logger.info("Saving expert weights:")
    target_model.save_compressed(args.outdir)

    model = AutoModelForCausalLM.from_pretrained(
            args.target_model, torch_dtype=torch.float16, trust_remote_code=True
    )


    logger.info("Saving base model:")
    sd = model.state_dict()
    to_remove = []
    for name in sd.keys():
        if name.startswith(target_model.layers_block_name):
            for inside_layer_module in target_model.inside_layer_modules:
                prefix, suffix = inside_layer_module.split(EXPERT_ID_PLACEHOLDER)
                if prefix in name and suffix in name and name.endswith(".weight"):
                    to_remove.append(name)

    for name in to_remove:
        del sd[name]

    model.save_pretrained(f"{args.outdir}/base/base_model.pt", state_dict=sd)
    logger.info("Saving base model finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
