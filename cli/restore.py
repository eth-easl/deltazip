import os
import torch
import json
import argparse
import transformers
from deltazip import AutoDeltaZipModelForCausalLM
from deltazip.utils.generate import merge
from huggingface_hub import snapshot_download 

def main(args):
    print(args)
    model_name = args.target_model.replace("/", ".")
    if not os.path.exists(args.target_model):
        args.target_model = snapshot_download(args.target_model)
    
    with open(os.path.join(args.target_model, "delta_config.json"), "r") as fp:
        config = json.load(fp)
        
    delta = AutoDeltaZipModelForCausalLM.from_compressed(args.target_model, device="cpu", strict=True, unpack=True)
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(config['base_model'], None)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model)
    
    delta = merge(base_model, delta)
    del base_model
    
    message = [{"role":"user", "content": "Who is Alan Turing?"}]
    prompt = tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt").cuda()
    outputs = delta.generate(prompt, max_length=128)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    del prompt
    model_name = "merged_" + model_name
    delta = delta.cpu()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    delta.save_pretrained(os.path.join(args.outdir, model_name))
    tokenizer.save_pretrained(os.path.join(args.outdir, model_name))
    print(f"[info] merged model saved to {os.path.join(args.outdir, model_name)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--outdir", type=str, default=".local/merged_models")
    args = parser.parse_args()
    main(args)
