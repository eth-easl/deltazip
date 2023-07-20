import os
import json
from transformers import AutoTokenizer
from src import BaseQuantizeConfig
def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)['text'] for line in fp.readlines()]
    quantize_config = BaseQuantizeConfig(
        bits = 2,
        group_size = 1024,
        sparsity=1.0
    )
    


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--task", type=str, default="answer_verification")
    parser.add_argument("--delta", type=str, default="facebook/opt-1.3b")