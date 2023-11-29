import os
import json
import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer, TextGenerationPipeline
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig


def postprocess(text):
    text = text.strip()
    # if starts with \n, take the remaining
    if text.startswith("\n"):
        text = text.split("\n")[1]
    # if there's \n left, take the first part
    text = text.split("\n")[0]
    return text


compress_config = BaseCompressionConfig(
    bits=4,
    group_size=128,
    sparsity=1,
    prunen=0,
    prunem=0,
    lossless="gdeflate",
    damp_percent=0.02,
)


def generate(args):
    print(args)
    if os.path.exists(args.output_file):
        logger.info(f"Output file {args.output_file} already exists, skipping...")
        return
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=args.fast_tokenizer
    )
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    tokenizer.skip_special_tokens = False
    with torch.inference_mode():
        base_model = AutoFMZipModelForCausalLM.from_pretrained(
            args.base_model, compress_config=compress_config
        )
        base_model = base_model.half()
        logger.info("Loading target model")
        if args.delta == "subtract":
            delta_model = AutoFMZipModelForCausalLM.from_compressed(
                args.target_model, strict=True, device="cpu", unpack=True
            )
        else:
            delta_model = AutoFMZipModelForCausalLM.from_pretrained(
                args.target_model, compress_config=compress_config
            )
        delta_model = delta_model.half()
        compressed_modules = []
        for x in base_model.inside_layer_modules:
            compressed_modules.extend(x)
        if args.delta == "subtract":
            for name, param in base_model.model.named_parameters():
                delta_model.model.state_dict()[name].copy_(
                    param + delta_model.model.state_dict()[name]
                )
        delta_model = delta_model.to(torch.device("cuda"))
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f][: args.n_samples]
        pipe = TextGenerationPipeline(
            model=delta_model, tokenizer=tokenizer, device="cuda"
        )
        logger.info("Pipeline Ready")
        prompts = [datum[args.input_field] for datum in data]
        outputs = pipe(
            prompts,
            max_new_tokens=args.max_length,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_full_text=False,
        )
        results = []

        for datum, output in zip(data, outputs):
            result = datum.copy()
            result["prediction"] = [o["generated_text"] for o in output]
            results.append(result)
        # create output dir if not exists
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))
        with open(args.output_file, "w") as f:
            for datum in results:
                f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--delta", type=str, default="")
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--input-field", type=str, default="input")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--fast-tokenizer", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()
    generate(args)
