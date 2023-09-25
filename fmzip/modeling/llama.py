import math
import torch
from ._const import *
from ._utils import *
import torch.nn as nn

from ._base import *
from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, apply_rotary_pos_emb, repeat_kv

def llama_mlp_forward(self, x):
    hidden_states = self.up_proj(x)
    gate_hiddent_states = self.gate_proj(x)
    hidden_states = self.act_fn(gate_hiddent_states * hidden_states)


def llama_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    base_query_states = self.q_proj(hidden_states)
    base_key_states = self.k_proj(hidden_states)
    base_value_states = self.v_proj(hidden_states)
    delta_query_states = [self.delta[i].q_proj(hidden_states[i]) for i in range(len(self.delta))]
    delta_key_states = [self.delta[i].k_proj(hidden_states[i]) for i in range(len(self.delta))]
    delta_value_states = [self.delta[i].v_proj(hidden_states[i]) for i in range(len(self.delta))]
    delta_query_states = torch.stack(delta_query_states, dim=0)
    delta_key_states = torch.stack(delta_key_states, dim=0)
    delta_value_states = torch.stack(delta_value_states, dim=0)
    query_states = base_query_states + delta_query_states
    key_states = base_key_states + delta_key_states
    value_states = base_value_states + delta_value_states

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    base_attn_output = self.o_proj(attn_output)
    delta_attn_output = [self.delta[i].o_proj(attn_output[i]) for i in range(len(self.delta))]
    delta_attn_output = torch.stack(delta_attn_output, dim=0)
    attn_output = base_attn_output + delta_attn_output

    if not output_attentions:
            attn_weights = None

    return attn_output, attn_weights, past_key_value

class LlamaFMZipForCausalLM(BaseFMZipModelForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel


__all__ = ["LlamaFMZipForCausalLM"]