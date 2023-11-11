import torch
import transformers
from typing import Optional, Tuple
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from ._base import *


def gpt_neox_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    has_layer_past = layer_past is not None
    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)
    qkv += torch.stack(
        [
            self.delta[i].query_key_value(hidden_states[i])
            for i in range(len(self.delta))
        ],
        dim=0,
    )

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # Cache QKV values
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    present = (key, value) if use_cache else None

    # Compute attention
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_size
    )

    main_attn_output = self.dense(attn_output)

    main_attn_output += torch.stack(
        [self.delta[i].dense(attn_output[i]) for i in range(len(self.delta))], dim=0
    )
    outputs = (main_attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs
