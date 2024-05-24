import accelerate
import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig, base_generation_strategies, modelling_gpt_neox_moe
from deltazip.modeling._const import EXPERT_ID_PLACEHOLDER
from loguru import logger
from safetensors.torch import save_file
import safetensors
from transformers import GPTNeoXConfig


def main(args):
    print(args)
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
    if args.target_model == "gpt_neox_moe":
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            args.tokenizer, use_fast=args.fast_tokenizer
        )
        with open(f"{args.model_path}/config.json", "r") as fp:
            config = GPTNeoXConfig(**json.load(fp))
        with accelerate.init_empty_weights():
            model = modelling_gpt_neox_moe.GPTNeoXForCausalLM(config)
        model = model.half()
        model = accelerate.load_checkpoint_and_dispatch(
            model, checkpoint=f"{args.model_path}/model.safetensors.index.json", device_map="auto", no_split_module_classes=['GPTNeoXLayer']
        )
        model.requires_grad_(False)
        target_model = AutoDeltaZipModelForCausalLM.from_model(
            model, compress_config=compress_config
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model, use_fast=args.fast_tokenizer
        )
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
    examples = [tokenizer(x, truncation=True) for x in examples]
    # examples = [e for e in examples if len(e['attention_mask']) != 0]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/base", exist_ok=True)
    
    logger.info("Saving base expert weights:")
    base_weights = target_model.get_moe_base_weights(base_generation_strategies.take_first)
    print(f"base_weights_keys: {base_weights.keys()}")
    save_file(base_weights, f"{args.outdir}/base/base_weights.safetensors")
    logger.info("Saving base weights finished")
    del base_weights
    
    target_model.lossy_compress(
        examples,
        batch_size=1,
        is_moe=True
    )
    # write to folder
    logger.info("Saving experts' delta weights:")
    target_model.save_compressed(args.outdir)

    if args.target_model == "gpt_neox_moe":
        model = modelling_gpt_neox_moe.GPTNeoXForCausalLM(config)
        model = model.half()
        files = os.listdir(args.model_path)
        files = [f for f in files if f.endswith("safetensors")]
        for f in files:
            print(f"Loading: {args.model_path}/{f}")
            safetensors.torch.load_model(model, f"{args.model_path}/{f}", strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model, torch_dtype=torch.float16, trust_remote_code=True
        )

    logger.info("Saving non-fc layers:")
    sd = model.state_dict()
    to_remove = []
    for name in sd.keys():
        if name.startswith(target_model.layers_block_name):
            for inside_layer_module in target_model.inside_layer_modules:
                prefix, suffix = inside_layer_module.split(EXPERT_ID_PLACEHOLDER)
                if prefix in name and suffix in name and name.endswith(".weight"):
                    to_remove.append(name)

    # Make sure we only save the non-fc layers (i.e the layers where MoE isn't applied)
    for name in to_remove:
        del sd[name]
    model.save_pretrained(f"{args.outdir}/base/base_model", state_dict=sd)
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
    parser.add_argument("--target-model", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--tokenizer", type=str)
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
    parser.add_argument("--shuffle-dataset", action="store_false")
    args = parser.parse_args()
    main(args)
