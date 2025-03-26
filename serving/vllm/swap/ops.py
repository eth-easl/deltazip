import torch
from typing import List, Tuple
import torch.nn.functional as F


def apply_swap_embed(
    x: torch.Tensor,
    packed_weights: List,
    indices: torch.Tensor,
    outputs: torch.Tensor,
):
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = F.embedding(inp, packed_weights[id])
        outputs[idx_mask] = output
    return outputs


def apply_swap_slice(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.LongTensor,
    y_offset: int,
    y_slice_size: int,
):
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = input[idx_mask]
        output_slice = F.linear(inp, weight[id])
        output[idx_mask, y_offset : y_offset + y_slice_size] += output_slice
    return output


def apply_swap_packed_nslice(
    x: torch.Tensor,
    stacked_weights: List,
    indices: torch.Tensor,
    output: torch.Tensor,
    output_slices: Tuple[int, ...],
):
    org_output = output
    x = x.view(-1, x.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        apply_swap_slice(
            output,
            x,
            stacked_weights[slice_idx],
            indices,
            offset_left,
            output_slices[slice_idx],
        )
        offset_left += output_slices[slice_idx]
    return output.view_as(org_output)


def apply_swap(
    x: torch.Tensor,
    stacked_weights: List,
    indices: torch.Tensor,
    outputs: torch.Tensor,
):
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = F.linear(inp, stacked_weights[id])
        outputs[idx_mask] += output
    return outputs


def apply_swap_logits(
    x: torch.Tensor,
    swap_weights: torch.Tensor,
    indices: torch.Tensor,
    base_output: torch.Tensor,
):
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        inp = x[indices == id]
        output = F.linear(inp, swap_weights[id])
        base_output[indices == id] += output
    return base_output
