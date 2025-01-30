import torch
import os
import json
from deltazip import AutoDeltaZipModelForCausalLM
import transformers

ignore_keywords = [
    'norm',
    'embed',
    'lm_head'
]

def merge(base, delta):
    base = base.bfloat16()
    delta = delta.bfloat16()
    print(f"base {base.device}")
    print(f"delta {delta.device}")
    for name, param in base.model.named_parameters():
        if any([kw in name for kw in ignore_keywords]):
            pass
        else:
            delta.model.state_dict()[name].copy_(
                param + delta.model.state_dict()[name]
            )
    delta = delta.to(torch.device("cuda"))
    return delta

def generate(target_model:str, prompt:str):
    
    with open(os.path.join(target_model, "delta_config.json"), "r") as fp:
        config = json.load(fp)
    model = AutoDeltaZipModelForCausalLM.from_compressed(target_model, device="cpu", strict=True, unpack=True)
    
    print(f"Loading base model {config['base_model']}...")
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(config['base_model'], None)
    
    model = merge(base_model, model)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['base_model'])
    pipe = transformers.TextGenerationPipeline(
        model=model, tokenizer=tokenizer, device="cuda"
    )
    outputs = pipe(
        [prompt],
        max_new_tokens=128,
        do_sample=True,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
        return_full_text=False,
    )[0][0]['generated_text']
    print(outputs)
    return outputs

def main(args):
    print(args)
    generate(args.target_model, args.prompt)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--prompt", type=str, default="Who is Alan Turing?")
    main(parser.parse_args())