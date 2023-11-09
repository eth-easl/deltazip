import torch
from typing import List
from fmzip.nn_modules.qlinear_cuda import QuantLinear
from fmzip.nn_modules.triton_utils.bmm import quant_bmm_248


def BatchedQuantLinearForward(inputs, layers: List[QuantLinear]):
    # assuming all bits are the same
    bits = layers[0].bits
    max_q = layers[0].maxq
    b_qweights = torch.stack([l.qweight for l in layers])
    b_qzeros = torch.stack([l.qzeros for l in layers])
    b_scales = torch.stack([l.scales for l in layers])
    b_g_idx = torch.stack([l.g_idx for l in layers])
    return quant_bmm_248(inputs, b_qweights, b_scales, b_qzeros, b_g_idx, bits, max_q)
