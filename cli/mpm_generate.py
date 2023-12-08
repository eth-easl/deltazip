import os
import json
import torch
import argparse
from loguru import logger
from timeit import default_timer as timer
from transformers import AutoTokenizer
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from deltazip.utils.delta_utils import xor_inverse, subtract_inverse
from deltazip.pipelines import MixedPrecisionModel


def postprocess(text):
    # logic:
    # if starts with \n, take the remaining
    if text.startswith("\n"):
        text = text.split("\n")[1]
    # if there's \n left, take the first part
    text = text.split("\n")[0]
    return text


def generate(args):
    print(args)
    mpm = MixedPrecisionModel(args.base_model)
    mpm.load_delta(args.target_model)
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]
    prompts = []
    for datum in data:
        prompts.append(datum)
    # process batch by batch
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        input_batch = [
            (prompt[args.input_field], args.target_model) for prompt in batch
        ]
        outputs = mpm.generate(
            input_batch,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        for j in range(len(outputs)):
            outputs[j] = outputs[j].replace(batch[j][args.input_field], "")
            batch[j]["prediction"] = [postprocess(outputs[j])]
        data[i : i + args.batch_size] = batch
    with open(args.output_file, "w") as f:
        for datum in data:
            f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--input-field", type=str, default="input")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    generate(args)
