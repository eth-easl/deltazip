import torch
from src.modules.opt import skip
from transformers import OPTForCausalLM

def get_opt_model(dtype=torch.float16):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=dtype)
    model.seqlen = model.config.max_position_embeddings
    return model

def get_opt_layers(model):
    return model.model.decoder.layers
