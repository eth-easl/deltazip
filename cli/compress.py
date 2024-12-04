import os
import json
import torch
import argparse
import datasets
import safetensors as st
from transformers import AutoTokenizer
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from deltazip.utils.generate import generate
from cli.utils import generate_readme, upload_and_delete

max_threads = str(min(8, os.cpu_count()))
os.environ['OMP_NUM_THREADS'] = max_threads
os.environ['OPENBLAS_NUM_THREADS'] = max_threads
os.environ['MKL_NUM_THREADS'] = max_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = max_threads
os.environ['NUMEXPR_NUM_THREADS'] = max_threads
os.environ['NUMEXPR_MAX_THREADS'] = max_threads

def get_max_sequence_length(config):
    if "max_position_embeddings" in config:
        return config.max_position_embeddings
    else:
        raise ValueError("Could not determine maximum sequence length from model configuration")

def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, use_fast=args.fast_tokenizer
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
        torch_dtype=torch.bfloat16,
    )
    if args.seq_len <0:
        args.seq_len = get_max_sequence_length(target_model.config)
        print(f"[info] set sequence length to {args.seq_len}")
        
    target_model = target_model.cuda()
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

    cal_ds = datasets.load_dataset(args.dataset, split=args.ds_split)
    cal_ds = cal_ds.shuffle(seed=42).select(range(args.n_samples))
    
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.seq_len,
            truncation=True,
            add_special_tokens=False,
        )    
        
    cal_ds = cal_ds.map(preprocess)
    examples = cal_ds.map(tokenize, remove_columns=cal_ds.column_names)
    
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
            if any([keyword in name for keyword in not_save_keywords]):
                print(f"[info] {name} is not saved")
                del target_model.state_dict()[name]

    if args.base_model != "" and args.delta != "":
        del base_model
    if args.prunen > 0 and args.prunem > 0:
        config_short = f"{args.bits}b_{args.prunen}n{args.prunem}m_{args.block_size}bs"
    else:
        config_short = f"{args.bits}b_{args.sparsity}sp_{args.block_size}bs"
    
    model_id = args.target_model.replace("/", ".") + f".{config_short}"
    outpath = os.path.join(args.outdir, model_id)
    
    target_model.save_compressed(outpath)
    tokenizer.save_pretrained(outpath)

    config_dict = {
        'base_model': args.base_model,
        'compress_config': compress_config.to_dict(),
        'target_modules': compressed_modules
    }
    with open(os.path.join(outpath, "delta_config.json"), "w") as fp:
        json.dump(config_dict, fp)

    # try:
    message = [{"role":"user", "content":args.test_prompt}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False)
    output = generate(outpath, prompt)
    
    readme = generate_readme({
        "model_id": args.base_model,
        "scheme": config_short,
        "dataset_id": args.dataset,
        "ds_split": args.ds_split,
        "seq_len": args.seq_len,
        "n_samples": args.n_samples,
        "prompt": prompt,
        "output": output
    })
    with open(os.path.join(outpath, "README.md"), "w") as f:
        f.write(readme)
    upload_and_delete(args.org_id, model_id, outpath)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to use for training, must be a path to a jsonl file.",
        default="HuggingFaceH4/ultrachat_200k"
    )
    parser.add_argument(
        "--ds-split",
        type=str,
        default="train_sft"
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
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument(
        "--lossless", type=str, default="none", choices=["gdeflate", "none"]
    )
    parser.add_argument("--delta", type=str, choices=["subtract", "xor"], default="")
    parser.add_argument("--sym", action="store_true", default=False)
    parser.add_argument("--desc-act", action="store_true")
    parser.add_argument("--large-model", action="store_true")
    parser.add_argument("--perc-damp", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default=".cache/compressed_models")
    parser.add_argument("--fast-tokenizer", action="store_true", default=True)
    parser.add_argument("--shuffle-dataset", action="store_true", default=True)
    parser.add_argument("--test-prompt", type=str, default="Who is Alan Turing?")
    parser.add_argument("--org-id", type=str, default="deltazip")
    args = parser.parse_args()
    main(args)
