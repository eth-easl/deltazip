import math
import torch
from ._const import *
from ._utils import *
import torch.nn as nn
from ._base import *
import transformers
from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
from typing import Optional, Tuple, List, Union
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from fmzip.utils.devices import get_gpu_count
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from loguru import logger

DEFAULT_CUDA_DEVICE = 1 if get_gpu_count() > 1 else 0
BASE_DEVICE = torch.device("cuda", DEFAULT_CUDA_DEVICE)


def llama_mlp_forward(self, x):
    hidden_states = self.up_proj(x.to(BASE_DEVICE, non_blocking=True))
    gate_hidden_states = self.gate_proj(x.to(BASE_DEVICE, non_blocking=True))
    up_xs = []
    gate_xs = []
    for i in range(len(self.delta)):
        if self.delta[i] is not None:
            with torch.cuda.stream(torch.cuda.Stream()):
                delta_x = x[i].to(self.delta[i].up_proj.qweight.device, non_blocking=True)
                up_x = self.delta[i].up_proj(delta_x)
                gate_x = self.delta[i].gate_proj(delta_x)
                up_xs.append(up_x)
                gate_xs.append(gate_x)
        else:
            up_xs.append(torch.zeros_like(hidden_states[i]))
            gate_xs.append(torch.zeros_like(gate_hidden_states[i]))

    hidden_states += torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in up_xs], dim=0
    )
    gate_hidden_states += torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in gate_xs], dim=0
    )

    hidden_states = self.act_fn(gate_hidden_states) * hidden_states
    base_hidden_states = hidden_states.to(BASE_DEVICE, non_blocking=True)
    base_down_hidden_states = self.down_proj(base_hidden_states)
    delta_down_hss = []

    for i in range(len(self.delta)):
        if self.delta[i] is not None:
            with torch.cuda.stream(torch.cuda.Stream()):
                delta_hidden_states = hidden_states[i].to(
                    self.delta[i].down_proj.qweight.device, non_blocking=True
                )
                delta_down_hidden_states = self.delta[i].down_proj(delta_hidden_states)
                delta_down_hss.append(delta_down_hidden_states)
        else:
            delta_down_hss.append(torch.zeros_like(base_down_hidden_states[i]))

    hidden_states = base_down_hidden_states + torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in delta_down_hss], dim=0
    )
    return hidden_states


def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    qs_deltas = []
    ks_deltas = []
    vs_deltas = []
    base_query_states = self.q_proj(hidden_states)
    base_key_states = self.k_proj(hidden_states)
    base_value_states = self.v_proj(hidden_states)

    for i in range(len(self.delta)):
        if self.delta[i] is not None:
            with torch.cuda.stream(torch.cuda.Stream()):
                delta_hidden_states = hidden_states[i].to(
                    self.delta[i].q_proj.qweight.device, non_blocking=True
                )
                qs_deltas.append(self.delta[i].q_proj(delta_hidden_states))
                ks_deltas.append(self.delta[i].k_proj(delta_hidden_states))
                vs_deltas.append(self.delta[i].v_proj(delta_hidden_states))
        else:
            qs_deltas.append(torch.zeros_like(base_query_states[i]))
            ks_deltas.append(torch.zeros_like(base_key_states[i]))
            vs_deltas.append(torch.zeros_like(base_value_states[i]))
    qs_delta_hidden_states = torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in qs_deltas], dim=0
    )
    ks_delta_hidden_states = torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in ks_deltas], dim=0
    )
    vs_delta_hidden_states = torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in vs_deltas], dim=0
    )
    query_states = base_query_states + qs_delta_hidden_states
    key_states = base_key_states + ks_delta_hidden_states
    value_states = base_value_states + vs_delta_hidden_states

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

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
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    base_attn_output = self.o_proj(attn_output.to(BASE_DEVICE, non_blocking=True))
    delta_attn_outputs = []
    for i in range(len(self.delta)):
        if self.delta[i] is not None:
            with torch.cuda.stream(torch.cuda.Stream()):
                delta_attn_output = self.delta[i].o_proj(
                    attn_output[i].to(
                        self.delta[i].o_proj.qweight.device, non_blocking=True
                    )
                )
        else:
            delta_attn_output = torch.zeros_like(base_attn_output[i])
        delta_attn_outputs.append(delta_attn_output)

    attn_output = base_attn_output + torch.stack(
        [x.to(BASE_DEVICE, non_blocking=True) for x in delta_attn_outputs], dim=0
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_rmsnorm_forward(self, hidden_states):
    # for other operations, add weights in base model to all deltas models
    outputs = []
    for i in range(len(self.delta)):
        if self.delta[i] is not None:
            with torch.cuda.stream(torch.cuda.Stream()):
                self.delta[i].weight += self.weight.to(self.delta[i].weight.device)
                hs = hidden_states[i]
                input_dtype = hs.dtype
                hs = hs.to(torch.float32)
                variance = hs.pow(2).mean(-1, keepdim=True)
                hs = hs * torch.rsqrt(variance + self.variance_epsilon)
                out = self.delta[i].weight * hs.to(input_dtype)
                outputs.append(out)
                self.delta[i].weight -= self.weight
        else:
            hs = hidden_states[i]
            input_dtype = hs.dtype
            hs = hs.to(torch.float32)
            variance = hs.pow(2).mean(-1, keepdim=True)
            hs = hs * torch.rsqrt(variance + self.variance_epsilon)
            out = self.weight * hs.to(input_dtype).to(BASE_DEVICE)
            outputs.append(out)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def llama_forcausallm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]
    logits = []
    for i in range(len(self.delta)):
        if self.delta[i] is not None:
            with torch.cuda.stream(torch.cuda.Stream()):
                self.delta[i].lm_head.weight += self.lm_head.weight.to(
                    self.delta[i].lm_head.weight.device
                )
                logit = self.delta[i].lm_head(hidden_states[i])
                self.delta[i].lm_head.weight -= self.lm_head.weight
        else:
            logit = self.lm_head(hidden_states[i].to(BASE_DEVICE))
        logits.append(logit)

    logits = torch.stack(logits, dim=0)
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        input_embeds = []
        for i in range(len(self.delta)):
            if self.delta[i] is not None:
                with torch.cuda.stream(torch.cuda.Stream()):
                    self.delta[i].embed_tokens.weight += self.embed_tokens.weight.to(
                        self.delta[i].embed_tokens.weight.device
                    )
                    input_embeds.append(self.delta[i].embed_tokens(input_ids[i]))
                    self.delta[i].embed_tokens.weight -= self.embed_tokens.weight
            else:
                input_embeds.append(self.embed_tokens(input_ids[i].to(BASE_DEVICE)))
        inputs_embeds = torch.stack(input_embeds, dim=0)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )
    hidden_states = inputs_embeds
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions)
                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class LlamaFMZipForCausalLM(BaseFMZipModelForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel


def parallelize_llama():
    transformers.models.llama.modeling_llama.LlamaMLP.forward = llama_mlp_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
        llama_attention_forward
    )
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = (
        llama_rmsnorm_forward
    )
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = (
        llama_forcausallm_forward
    )
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward


__all__ = ["LlamaFMZipForCausalLM"]
