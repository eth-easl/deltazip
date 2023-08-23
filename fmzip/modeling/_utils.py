from typing import Union

import torch
import transformers
import torch.nn as nn
from loguru import logger
from transformers import AutoConfig

from ._const import SUPPORTED_MODELS, CPU, CUDA_0
from ..utils.import_utils import dynamically_import_QuantLinear
from ..nn_modules.qlinear import QuantLinear
from ..utils.attr_utils import rsetattr


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to_device(obj: Union[torch.Tensor, nn.Module], device: torch.device):
    if get_device(obj) != device:
        obj = obj.to(device)
    return obj


def find_layers(module, layers=None, name=''):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_module_by_name(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


def make_quant(module, names, bits, group_size, name='', use_triton=False, use_cuda_fp16=True, desc_act=False):
    from ..nn_modules.qlinear import QuantLinear
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            ori_layer_device = get_device(getattr(module, attr))
            delattr(module, attr)
            if type(tmp) == nn.Linear:
                in_features = tmp.in_features
                out_features = tmp.out_features
            elif type(tmp) == nn.Conv2d:
                in_features = tmp.in_channels
                out_features = tmp.out_channels
            elif type(tmp) == transformers.pytorch_utils.Conv1D:            
                in_features = tmp.weight.shape[0]
                out_features = tmp.weight.shape[1]
            # if bits is a dict
            if isinstance(bits, dict):
                bits = bits[name1]
            new_layer = QuantLinear(bits, group_size, in_features, out_features, tmp.bias is not None)
            new_layer.device = ori_layer_device
            setattr(module, attr, new_layer.to(ori_layer_device))
    for name1, child in module.named_children():
        make_quant(child, names, bits, group_size, name + '.' + name1 if name != '' else name1, use_triton=use_triton, use_cuda_fp16=use_cuda_fp16,desc_act=desc_act)

def unpack_model(model):
    logger.info('Unpacking model...')
    layers = find_layers(model, layers=[QuantLinear])
    for name in layers:
        rsetattr(model, name, layers[name].unpack())
    logger.info('Model unpacked.')

def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    warmup_triton: bool = False,
    force_layer_back_to_cpu: bool = False
):
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size)

    if force_layer_back_to_cpu:
        model.to(CPU)

    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size, use_triton=use_triton, use_cuda_fp16=use_cuda_fp16, desc_act=desc_act)
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        logger.info(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx = layers[name].to(CPU), scale.to(CPU), zero.to(CPU), g_idx.to(CPU)
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        qlayers[name].to(layer_device)
    logger.info('Model packed.')
    # after packing, we further apply lossless compression

def check_and_get_model_type(model_dir):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


__all__ = [
    "get_device",
    "move_to_device",
    "find_layers",
    "get_module_by_name",
    "make_quant",
    "pack_model",
    "check_and_get_model_type"
]