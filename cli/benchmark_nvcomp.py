import torch
from kvikio.nvcomp import GdeflateManager as manager
from safetensors import safe_open
from safetensors.numpy import save_file
import numpy as np
import cupy as cp
import time
comp_manager = manager()

# tensors = {}
# with safe_open(".cache/model_delta.safetensors", framework="pt", device="cpu") as f:
#    for key in f.keys():
#         compressed_tensor = comp_manager.compress(cp.array(f.get_tensor(key).numpy()))
#         tensors[key] = cp.asnumpy(compressed_tensor)

# # write to file with safetensors
# save_file(tensors, ".cache/model_delta.fmzip.safetensors")

# read from file with safetensors as a cupy array

tensors = {}
with safe_open(".cache/model_delta.fmzip.safetensors", framework="np", device="cpu") as f:
    for key in f.keys():
        tensors[key] = comp_manager.decompress(cp.array(f.get_tensor(key)))
        time.sleep(20)