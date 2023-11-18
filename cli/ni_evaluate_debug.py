import os
import json
import torch
import argparse
from loguru import logger
from timeit import default_timer as timer
from transformers import AutoTokenizer, TextGenerationPipeline
from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig
from fmzip.utils.delta_utils import xor_inverse, subtract_inverse


def postprocess(text):
    # logic:
    # remove leading and trailing spaces
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

raw_model = "/mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/open_llama_3b_v2/task372_synthetic_palindrome_numbers/global_step105/"
raw_model = AutoFMZipModelForCausalLM.from_pretrained(
    raw_model, compress_config=compress_config
)
raw_model = raw_model.half()
raw_model = raw_model.to(torch.device("cuda"))
lm_head = torch.nn.Parameter(raw_model.state_dict()["lm_head.weight"].cuda().half())
embed_token = torch.nn.Parameter(
    raw_model.state_dict()["model.embed_tokens.weight"].cuda().half()
)


def generate(args):
    print(args)
    # just placeholder, we don't need it for base model...
    # (todo:xiaozhe) remove the need of compress_config

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    tokenizer.skip_special_tokens = False
    with torch.inference_mode():
        base_model = AutoFMZipModelForCausalLM.from_pretrained(
            args.base_model, compress_config=compress_config
        )
        base_model = base_model.half()
        base_model = base_model.to(torch.device("cuda"))
        logger.info("Loading target model")
        delta_model = AutoFMZipModelForCausalLM.from_compressed(
            args.target_model, strict=True, device="cpu", unpack=True
        )
        delta_model = delta_model.half()
        delta_model = delta_model.to(torch.device("cuda"))
        if args.delta == "subtract":
            print(f"Subtracting")
            for name, param in base_model.named_parameters():
                if "layernorm" not in name:
                    delta_model.state_dict()[name].copy_(
                        param + delta_model.state_dict()[name]
                    )
        elif args.delta == "xor":
            raise NotImplementedError
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f][:10]
        torch.cuda.empty_cache()
        # patch delta model
        delta_model.lm_head.weight = lm_head
        delta_model.model.embed_tokens.weight = embed_token

        for name, param in delta_model.named_parameters():
            print(f"{name}, {torch.max(param - raw_model.state_dict()[name])}")
        torch.cuda.synchronize()
        pipe = TextGenerationPipeline(
            model=delta_model, tokenizer=tokenizer, device="cuda"
        )
        logger.info("Pipeline Ready")
        prompts = [datum[args.input_field] + "\n" for datum in data]
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
            result["prediction"] = [postprocess(o["generated_text"]) for o in output]
            result["raw_prediction"] = [o["generated_text"] for o in output]
            results.append(result)
        for result in results:
            print(f"output: {result['output']} prediction: {result['prediction']}")
        # with open(args.output_file, "w") as f:
        #     for datum in data:
        #         f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--target-model", type=str, default="gpt2")
    parser.add_argument("--delta", type=str, default="")
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--input-field", type=str, default="input")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    generate(args)
