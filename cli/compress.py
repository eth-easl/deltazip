import os
import json
import torch
import argparse
import safetensors as st
from transformers import AutoTokenizer
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
import os
import math

max_threads = str(min(8, os.cpu_count()))
os.environ['OMP_NUM_THREADS'] = max_threads
os.environ['OPENBLAS_NUM_THREADS'] = max_threads
os.environ['MKL_NUM_THREADS'] = max_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = max_threads
os.environ['NUMEXPR_NUM_THREADS'] = max_threads
os.environ['NUMEXPR_MAX_THREADS'] = max_threads

def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=args.fast_tokenizer
    )
    compress_config = BaseCompressionConfig(
        bits=args.bits,
        sparsity=args.sparsity,
        block_size=args.block_size,
        prunen=args.prunen,
        prunem=args.prunem,
        lossless=args.lossless,
        damp_percent=args.perc_damp,
        desc_act=args.desc_act,
        sym=args.sym,
    )
    print("[info] compress config:", compress_config)
    target_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        args.target_model, 
        compress_config=compress_config,
        torch_dtype=torch.float16,
        # max_memory = {
        #     0: "60GIB", 
        #     1: "60GIB",
        #     2: "60GIB", 
        #     3: "60GIB", 
        #     4: "60GIB", 
        #     5: "60GIB", 
        #     6: "60GIB", 
        #     7: "60GIB", 
        #     "cpu": "140GIB"
        # }
    )
    ignore_keywords = [
        'norm',
        'embed',
        'lm_head'
    ]
    not_save_keywords = [
        'norm',
    ]
    target_model.requires_grad_(False)
    if args.base_model != "" and args.delta != "":
        print("[info] base model is defined, delta mode enabled")
        base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
            args.base_model,
            compress_config=compress_config,
            torch_dtype=torch.float16,
        )
        base_model.requires_grad_(False)
    torch.cuda.empty_cache()
    # now time to prepare inspect dataset
    with open(args.dataset, "r") as fp:
        examples = [json.loads(line)["text"] for line in fp.readlines()]
    if args.n_samples <= 0:
        examples = examples
    else:
        if args.shuffle_dataset:
            import random
            random.seed(42)
            random.shuffle(examples)
        examples = examples[: args.n_samples]
    examples = [tokenizer(x) for x in examples]
    if args.base_model != "" and args.delta != "":
        target_model.lossy_compress(
            examples,
            batch_size=1,
            base_model=base_model,
        )
    else:
        target_model.lossy_compress(
            examples,
            batch_size=1,
        )
    # write to folder
    os.makedirs(args.outdir, exist_ok=True)
    # for weights that are not compressed, we calculate delta afterward compression
    if args.large_model:
        # for large models - save a temporary results to avoid re-run
        tensors = {}
        for name, param in target_model.named_parameters():
            if not param.is_meta:
                tensors[name] = param.data.cpu().clone().detach()
        st.torch.save_file(
            tensors, os.path.join(args.outdir, "temp.safetensors")
        )
        target_model_ref = AutoDeltaZipModelForCausalLM.from_pretrained(
            args.target_model, 
            compress_config=compress_config,
            torch_dtype=torch.float16,
        )
        missing_state_dict = target_model_ref.state_dict()
        missing_state_dict = {
            k: v for k, v in missing_state_dict.items() if k not in tensors
        }
        print(f"[info] loaded keys: {missing_state_dict.keys()}")
        missing_key, unexpected_key = target_model.load_state_dict(missing_state_dict, strict = False, assign=True)
        print(f"[info] missing keys: {missing_key}")
        print(f"[info] unexpected keys: {unexpected_key}")
        for name, param in target_model.named_parameters():
            if param.is_meta:
                print(f"[info] {name} is on meta")
        del target_model_ref
    
    if args.base_model != "" and args.delta != "":
        compressed_modules = []
        for x in base_model.inside_layer_modules:
            compressed_modules.extend(x)
        for name, param in target_model.named_parameters():
            # if all([module not in name for module in compressed_modules]):
            #     print(f"[info] {name} is compressed, saving in full...")
                
            #     target_model.state_dict()[name] = param
            # else:
            #     print(f"[info] {name} is not compressed, saving in full...")
            #     target_model.state_dict()[name] = param
            if any([keyword in name for keyword in not_save_keywords]):
                print(f"[info] {name} is not saved")
                del target_model.state_dict()[name]
            # if "bias" in name or all(
            #     [modules not in name for modules in compressed_modules]
            # ):
                # base_weight = base_model.state_dict()[name]
                # if base_weight.device != param.device:
                #     base_weight = base_weight.to(param.device)
                # target_model.state_dict()[name] = param - base_weight
                
    if args.base_model != "" and args.delta != "":
        del base_model
    # run a forward pass to make sure the model is working
    target_model.save_compressed(args.outdir)
    with open(os.path.join(args.outdir, "compressed_modules.json"), "w") as fp:
        json.dump(compressed_modules, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        default="answer_verification",
        help="The dataset to use for training, must be a path to a jsonl file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="How many data samples used for calibration, -1 means all.",
    )
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument(
        "--lossless", type=str, default="gdeflate", choices=["gdeflate"]
    )
    parser.add_argument("--delta", type=str, choices=["subtract", "xor"], default="")
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--desc-act", action="store_true")
    parser.add_argument("--large-model", action="store_true")
    parser.add_argument("--perc-damp", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default=".cache/compressed_models")
    parser.add_argument("--fast-tokenizer", action="store_true")
    parser.add_argument("--shuffle-dataset", action="store_true")
    args = parser.parse_args()
    main(args)
