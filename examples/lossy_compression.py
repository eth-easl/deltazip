import os
import json
from transformers import AutoTokenizer
from fmzip import BaseCompressionConfig, AutoFMZipModelForCausalLM

def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)["text"] for line in fp.readlines()]
    compress_config = BaseCompressionConfig(
        bits=args.bits,
        group_size=args.group_size,
        sparsity=args.sparsity,
        prunen=0,
        prunem=0,
        lossless='gdeflate'
    )
    examples = [tokenizer(x) for x in examples]
    model = AutoFMZipModelForCausalLM.from_pretrained(
        args.target_model, compress_config=compress_config
    )
    model.lossy_compress(examples)
    directory_name = f"{args.target_model.replace('/','.')}-{args.bits}bit-{args.group_size}g-{args.sparsity}sparsity"
    base_directory = os.path.join(".cache/compressed_models/", directory_name)
    os.makedirs(base_directory, exist_ok=True)
    model.save_compressed(
        os.path.join(base_directory, directory_name)
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset", type=str, default="answer_verification")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=1024)

    args = parser.parse_args()
    main(args)
