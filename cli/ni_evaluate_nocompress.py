import os
import json
import torch
import argparse
from loguru import logger
from timeit import default_timer as timer
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse

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
    # just placeholder, we don't need it for base model...
    # (todo:xiaozhe) remove the need of compress_config
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        target_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.bfloat16)
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f]
        # to leave more memory for higher-throughput generation,
        # put the base model to cpu
        torch.cuda.empty_cache()
        pipe = TextGenerationPipeline(
            model=target_model, tokenizer=tokenizer, device="cuda"
        )
        logger.info("Pipeline Ready")
        prompts = [datum[args.input_field] for datum in data]
        outputs = pipe(
            prompts,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_full_text=False,
        )
        results = []
        for datum, output in zip(data, outputs):
            result = datum
            result["prediction"] = [postprocess(o["generated_text"]) for o in output]
            results.append(result)
        with open(args.output_file, "w") as f:
            for datum in data:
                f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--input-field", type=str, default="input")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    generate(args)
