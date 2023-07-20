import os
import json
from transformers import AutoTokenizer
from src import BaseQuantizeConfig, AutoFMZipModelForCausalLM

def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)['text'] for line in fp.readlines()]
    quantize_config = BaseQuantizeConfig(
        bits = 2,
        group_size = 1024,
        sparsity=0
    )
    examples = [
        tokenizer(x) for x in examples
    ]
    target_model = AutoFMZipModelForCausalLM.from_pretrained(args.target_model, quantize_config)
    target_model.quantize(examples)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-1.3b", required=False)
    parser.add_argument("--dataset", type=str, default="answer_verification")
    parser.add_argument("--target-model", type=str, default="facebook/opt-1.3b")
    args = parser.parse_args()
    main(args)