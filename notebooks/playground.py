import json
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from numpy import count_nonzero
# from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig


take_first = lambda x: x[0]

def get_module_by_name(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module
  
def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

        
def calculate_sparsity(tensor):
    return 1.0 - (count_nonzero(tensor) / float(tensor.size))

# compress_config = BaseCompressionConfig()
# delta_model = AutoDeltaZipModelForCausalLM.from_pretrained("mlabonne/phixtral-2x2_8", compress_config)
# for k,v in delta_model.model.named_modules():
    # print(k)
# print(delta_model.model.config.__dict__)
# layers = get_module_by_name(delta_model.model, delta_model.layers_block_name)
# # print(layers)
# full = find_layers(layers[0])
# print(full)
# weights = torch.load("/mnt/scratch/bborisov/.cache/compressed_phixtral_2x2/base_weights.pt")
# print(weights)

# sd = model.state_dict()
# keys = [x for x in sd.keys() if x.startswith("transformer.h")]
# inside_layer_modules = [f"moe.mlp.EXPERT_ID.fc1", f"moe.mlp.EXPERT_ID.fc2"]
# print(keys)
# to_remove = set()
# for key in keys:
#     for inside_layer_module in inside_layer_modules:
#         prefix, suffix = inside_layer_module.split("EXPERT_ID")
#         # print(f"prefix {prefix} | suffix {suffix}")
#         if suffix in key and suffix in key and key.endswith(".weight"):
#             print (f"FOUND KEY TO REMOVE {key}")
#             to_remove.add(key)
# for k in to_remove:
#     sd[k] = torch.randn(0)

# print(sd.keys())
# model.load_state_dict(sd, strict=False)
# model.modules
# for name, parameter in model.named_parameters():
#     # print(name)
#     # print(parameter)
#     if name.startswith("transformer.h"):
#         for inside_layer_module in inside_layer_modules:
#             prefix, suffix = inside_layer_module.split("EXPERT_ID")
#             if suffix in name and suffix in name and name.endswith(".weight"):
#                 parameter.data = torch.randn(0)

# print("------ After changes: -------")
# for name, parameter in model.named_parameters():
#     if name.startswith("transformer.h"):
#         for inside_layer_module in inside_layer_modules:
#             prefix, suffix = inside_layer_module.split("EXPERT_ID")
#             if suffix in name and suffix in name and name.endswith(".weight"):
#                 print(f"{name}: {parameter.data.size()}")

# torch.save(model, "/mnt/scratch/bborisov/deltazip/notebooks/temp.pt")
# model = torch.load( "/mnt/scratch/bborisov/deltazip/notebooks/temp.pt")
# print("----- After reloading: ------")
# sd = model.state_dict()
# to_remove = []
# for name in sd.keys():
#     if name.startswith("transformer.h"):
#         for inside_layer_module in inside_layer_modules:
#             prefix, suffix = inside_layer_module.split("EXPERT_ID")
#             if suffix in name and suffix in name and name.endswith(".weight"):
#                 to_remove.append(name)

# for name in to_remove:
#     del sd[name]

# print(sd.keys())

# ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
# ds = ds["train_sft"]
# print(ds)
# print(type(ds["prompt"]))
# prompts = [{"text": prompt} for prompt in ds["prompt"]]
# with open("stf.jsonl", "w") as fp:
#     for prompt in prompts:
#         fp.write(json.dumps(prompt))
#         fp.write("\n")

model = transformers.AutoModelForCausalLM.from_pretrained(f"mlabonne/phixtral-2x2_8", trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained("mlabonne/phixtral-2x2_8")
activation_dicts = []
def wrapper(name):
    d = {}
    activation_dicts.append(d)
    def func(module, x, y):
        x = x[0]
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = module.gate(x)
        expert_weights, expert_indices = torch.topk(scores, module.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        print(f"scores, {scores}")
        print(f"expert_indices: {expert_indices[0].detach()}")
        print(f"expert_weights: {expert_weights[0].detach()}")
        d[name] = {
            expert_indices[0][i].item(): expert_weights[0][i].item() for i in range(module.num_experts_per_tok)
        }

    return func


for name, module in model.named_modules():
    print(name)
    if name.endswith("moe"):
        module.register_forward_hook(wrapper(name), always_call=True)
pipe = transformers.TextGenerationPipeline(
    model=model, tokenizer=tokenizer, device="cuda"
)
print(pipe(
"How does one calculate the area of a circle?",            
max_new_tokens=128,
do_sample=True,
temperature=0.6,
top_k=50,
top_p=0.9,
return_full_text=False,
)[0]['generated_text'])
print(activation_dicts)