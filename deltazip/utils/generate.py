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
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(target_model)
    message = [{"role":"user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt").cuda()
    outputs = model.generate(prompt, max_length=128)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(outputs)
    return outputs