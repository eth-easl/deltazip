import torch.nn as nn
import transformers
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from numpy import count_nonzero
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig

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

compress_config = BaseCompressionConfig()
# model = AutoModelForCausalLM.from_pretrained(f"mlabonne/phixtral-2x2_8", trust_remote_code=True)
delta_model = AutoDeltaZipModelForCausalLM.from_pretrained("mlabonne/phixtral-2x2_8", compress_config)

layers = get_module_by_name(delta_model.model, delta_model.layers_block_name)
# print(layers)
full = find_layers(layers[0])
print(full)
# for names in delta_model.inside_layet_modules:
