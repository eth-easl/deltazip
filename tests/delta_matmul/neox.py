import torch
import transformers
from typing import Optional, Tuple
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXConfig,
    apply_rotary_pos_emb,
)
from fmzip.modeling.gpt_neox import parallelize_neox


def forward(
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
    attn_output = self.dense(attn_output)
    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention.forward = forward

config = GPTNeoXConfig(
    vocab_size=50432,
    hidden_size=6144,
    num_layers=2,
)

hidden_states = torch.rand(1, 1024, 6144)
attention_masks = torch.ones(1, 1024)
position_ids = torch.arange(1024).unsqueeze(0)

base_attention = GPTNeoXAttention(config)
base_qkv_weights = base_attention.query_key_value.weight
base_dense = base_attention.dense.weight

delta_qkv_weight = torch.rand_like(base_qkv_weights)
delta_dense_weight = torch.rand_like(base_dense)
delta_attention = GPTNeoXAttention(config)
delta_attention.query_key_value.weight = torch.nn.Parameter(delta_qkv_weight)
delta_attention.dense.weight = torch.nn.Parameter(delta_dense_weight)

ft_attention = GPTNeoXAttention(config)
ft_attention.query_key_value.weight = torch.nn.Parameter(
    base_qkv_weights + delta_qkv_weight
)
ft_attention.dense.weight = torch.nn.Parameter(base_dense + delta_dense_weight)

ft_attention_output, _ = ft_attention(hidden_states, attention_masks, position_ids)
print(ft_attention_output)
# now parallelize the attention forward
parallelize_neox()

setattr(base_attention, "delta", [delta_attention])
parallel_attention_output, _ = base_attention(
    hidden_states, attention_masks, position_ids
)

print(parallel_attention_output)
print(torch.allclose(ft_attention_output, parallel_attention_output))
