import os
import json
from transformers import AutoTokenizer
from src import BaseCompressionConfig, AutoFMZipModelForCausalLM

def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)['text'] for line in fp.readlines()]
    compress_config = BaseCompressionConfig(
        bits = 4,
        group_size = 1024,
        sparsity=0.1
    )
    examples = [
        tokenizer(x) for x in examples
    ]
    model = AutoFMZipModelForCausalLM.from_pretrained(args.target_model, compress_config=compress_config)
    model.lossy_compress(examples)
    model.save_compressed("/home/xzyao/Documents/cache/compressed_models/")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dataset", type=str, default="answer_verification")
    parser.add_argument("--target-model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()
    main(args)
