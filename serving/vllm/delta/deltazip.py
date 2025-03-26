import os
import torch
from typing import Optional, Tuple, List, Any
import torch.nn.functional as F

BITWIDTH = int(os.environ.get("BITWIDTH", "4"))
USE_BITBLAS = os.environ.get("USE_BITBLAS", "0") == "1"
USE_MARLIN = os.environ.get("USE_MARLIN", "0") == "1"

from triteia.ao.ops.ibmm.ibmm_marlin import (
    ibmm_sparse_marlin as quant_select_bmm_248,
)

def add_delta(
    y: torch.Tensor,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    indices: torch.LongTensor,
    meta: Optional[torch.Tensor] = None,
):
    """
    semantics:
        y[i] += (
            x[i].unsqueeze(0)
            @ qweight[indices[i], :, :].transpose(-1, -2)
        ).squeeze(0)
    """
    if not USE_BITBLAS:
        g_idx = g_idx.repeat(qweight.shape[0], 1)
        g_idx = g_idx.to(qweight.device)
    quant_select_bmm_248(
        BITWIDTH, indices, meta, y, x, qweight, qzeros, scales, g_idx, bias=None
    )
    return y


def add_delta_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    indices: torch.LongTensor,
    scale: float,
    y_offset: int,
    y_slice_size: int,
    *,
    buffer: Optional[torch.Tensor] = None,
    meta: Optional[torch.Tensor] = None,
):
    """
    semantics:
        y[i] += (
            x[i].unsqueeze(0)
            @ qweight[indices[i], :, :].transpose(-1, -2)
        ).squeeze(0)
    """
    if not USE_BITBLAS:
        g_idx = g_idx.repeat(qweight.shape[0], 1)
        g_idx = g_idx.to(qweight.device)
    quant_select_bmm_248(
        BITWIDTH,
        indices,
        meta,
        y[:, y_offset : y_offset + y_slice_size],
        x,
        qweight,
        qzeros,
        scales,
        g_idx,
        bias=None,
    )
    return y


def apply_delta(
    x: torch.Tensor,
    qweight_stacked: torch.Tensor,
    qzeros_stacked: torch.Tensor,
    scales_stacked: torch.Tensor,
    g_idx_stacked: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
):
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    add_delta(
        output,
        x,
        qweight_stacked,
        qzeros_stacked,
        scales_stacked,
        g_idx_stacked,
        indices,
    )
    return output.view_as(org_output)


def apply_delta_packed_nslice(
    x: torch.Tensor,
    qweight_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qzeros_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scales_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    g_idx_stacked: List[torch.Tensor],
    indices: torch.Tensor,
    output: torch.Tensor,
    output_slices: Tuple[int, ...],
):
    """
    Applies delta to each input.
    This method applies all deltas to each input. It uses the
    indices vector to determine which delta yields the
    correct output. An index of -1 means no delta should be
    applied. This method adds the final delta results to the
    output.

    This method is used for layers that are composed of multiple sublayers
    (slices) packed together.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        qweight_stacked:    3 element tuple of (num_deltas, 1,  hidden_dim/pack_factor, hidden_dim)
        qzeros_stacked:     3 element tuple of (num_deltas, 1, 1, hidden_dim/pack_factor)
        indices:           (batch_size)
        output:            (batch_size, q_slice_size + 2*kv_slice_size)
        output_slices:     n-1 element tuple of (slice_size...),
                           where n is number of slices
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        add_delta_slice(
            output,
            x,
            qweight_stacked[slice_idx],
            qzeros_stacked[slice_idx],
            scales_stacked[slice_idx],
            g_idx_stacked[slice_idx],
            indices,
            1.0,
            offset_left,
            output_slices[slice_idx],
        )
        offset_left += output_slices[slice_idx]
    return output.view_as(org_output)


def apply_delta_uncompressed(
    x: torch.Tensor,
    delta_weights: torch.Tensor,
    indices: torch.Tensor,
    base_output: torch.Tensor,
):
    """
    Applies delta to each input.

    This method applies all deltas to each input. An index of -1 means no delta should be applied.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        weights:           ()
        indices:           (batch_size)
        output:            (batch_size, hidden_dim)
    """
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        with torch.cuda.stream(torch.cuda.Stream()):
            inp = x[indices == id]
            output = F.linear(inp, delta_weights[id])
            base_output[indices == id] += output
    torch.cuda.synchronize()
    return base_output


def apply_delta_embed(
    x: torch.Tensor,
    delta_weights: List,
    indices: torch.Tensor,
    base_output: torch.Tensor,
):
    """
    Applies delta to each input.
    This method applies all deltas to each input. It uses the
    indices vector to determine which delta yields the
    correct output. An index of -1 means no delta should be
    applied. This method adds the final delta results to the
    output.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        delta_weights:     list of delta weights
        indices:           (batch_size)
    """
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        with torch.cuda.stream(torch.cuda.Stream()):
            idx_mask = indices == id
            inp = x[idx_mask]
            output = F.embedding(inp, delta_weights[id])
            base_output[idx_mask] += output
    torch.cuda.synchronize()
    return base_output
