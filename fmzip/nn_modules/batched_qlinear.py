import math
import torch
from typing import List
from loguru import logger
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


def WarmupBQLForward(models: List, b=8, seqlen=2048):
    """
    ideally we should also support cuda graphs
    (todo: xiaozhe; high priority)
    """
    from tqdm import tqdm

    kn_values = {}
    for model in models:
        for _, m in model.named_modules():
            if not isinstance(m, QuantLinear):
                continue
            k = m.infeatures
            n = m.outfeatures
            if (k, n) not in kn_values:
                kn_values[(k, n)] = [m]
    logger.info(f"Found {len(kn_values)} unique KN Linear values.")
    with torch.no_grad():
        for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
            m = 2**m
            for (k, n), (
                qweight,
                scales,
                qzeros,
                g_idx,
                bits,
                maxq,
            ) in kn_values.items():
                a = torch.randn(b, m, k, dtype=torch.float16, device=model.device)
            quant_bmm_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
    del kn_values
