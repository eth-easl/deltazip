import torch
from deltazip import AutoDeltaZipModelForCausalLM

ignore_keywords = [
    'norm',
    'embed',
    'lm_head'
]

def merge(base, delta):
    base = base.bfloat16().cpu()
    delta = delta.bfloat16().cpu()
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

def generate(model, base_model, tokenizer, prompt:str):
    model = merge(base_model, model)
    message = [{"role":"user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt").cuda()
    outputs = model.generate(prompt, max_length=128)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return message, outputs