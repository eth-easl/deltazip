import os
import json
import copy
import torch
import argparse
from transformers import AutoTokenizer
from src import BaseQuantizeConfig, AutoGPTQForCausalLM

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)['text'] for line in fp.readlines()]
    examples = examples[:args.n_samples]
    examples = [
        tokenizer(x) for x in examples
    ]
    quantize_config = BaseQuantizeConfig(
        bits=args.wbit,
        group_size=args.group_size,
    )

    base_model  = AutoGPTQForCausalLM.from_pretrained(args.base_model, quantize_config)
    target_model = AutoGPTQForCausalLM.from_pretrained(args.target_model, quantize_config)
    original_target_model = copy.deepcopy(target_model)
    # now find the delta model
    # don't rely on zip() because the order of the parameters may change (?)
    for name1, param1 in base_model.named_parameters():
        target_model.state_dict()[name1] -= param1
    base_model.requires_grad_(False)

    # now check if the delta model is correct
    with torch.no_grad():
        for name1, param1 in base_model.named_parameters():
            target_param = original_target_model.state_dict()[name1]
            delta_param = target_model.state_dict()[name1]
            assert torch.allclose(target_param, param1+delta_param, atol=1e-7)

    # now quantize the delta model
    target_model.quantize(examples)

    output_dir = os.path.join(args.out_dir, f"{args.target_model.split('/')[-1]}-{args.wbit}bit-{args.group_size}g-{args.sparsity}s-delta")

    target_model.save_quantized(output_dir, use_safetensors=True)
    logging.info(f"Quantized model saved to {output_dir}")

if __name__=="__main__":
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--target-model', type=str, default='.cache/models/answer_verification')
    parser.add_argument('--dataset', type=str, default='.cache/ni_calib/train/answer_verification.jsonl')
    parser.add_argument('--wbit', type=int, default=2, choices=[2,3,4,8])
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='.cache/compressed_models/')
    parser.add_argument('--group-size', type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main(args)