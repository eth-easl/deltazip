import torch
from safetensors import safe_open


tensors = {}
with safe_open(".cache/compressed_models/answer_verification-2bit-1024g-0.95s-delta/gptq_model-2bit-1024g.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

for k in tensors.keys():
    print(k, tensors[k].shape)
print(tensors['model.decoder.layers.9.self_attn.k_proj.qzeros'])