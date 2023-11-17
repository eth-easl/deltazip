"""
Given two models, base and target, we want to compute the delta between them.
"""
import torch
from ..modeling import AutoFMZipModelForCausalLM


def calculate_sparsity(tensor: torch.Tensor):
    return (tensor == 0).sum().item() / tensor.numel()


def subtract(base: AutoFMZipModelForCausalLM, target: AutoFMZipModelForCausalLM):
    with torch.no_grad():
        for name, param in target.named_parameters():
            param.copy_(20 * (param - base.state_dict()[name]))
    return target


def xor(base: AutoFMZipModelForCausalLM, target: AutoFMZipModelForCausalLM):
    with torch.no_grad():
        for name, param in target.named_parameters():
            param ^= base.state_dict()[name]
    return target


def subtract_inverse(base: AutoFMZipModelForCausalLM, delta: AutoFMZipModelForCausalLM):
    with torch.no_grad():
        for name, param in delta.named_parameters():
            param.copy_(param + base.state_dict()[name])
    return delta


def xor_inverse(base: AutoFMZipModelForCausalLM, delta: AutoFMZipModelForCausalLM):
    with torch.no_grad():
        for name, param in delta.named_parameters():
            param ^= base.state_dict()[name]
    return delta
