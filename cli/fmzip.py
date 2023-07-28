import argparse
from transformers import AutoTokenizer
from fmzip import AutoFMZipModelForCausalLM
def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    base_model = AutoFM

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset", type=str, default="answer_verification", hint="The dataset to use for training, must be a path to a jsonl file.")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--lossless", type=str, default="gdeflate", choices=['gdeflate'])
    parser.add_argument("--delta", type=str, choices=['subtract', 'xor'], default='subtract')
    args = parser.parse_args()
    main(args)