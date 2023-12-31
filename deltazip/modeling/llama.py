from ._const import *
from ._utils import *
from ._base import *
import transformers

from deltazip.modeling.llama_monkey_patch import (
    llama_attention_forward,
    llama_mlp_forward,
    llama_model_forward,
    llama_rmsnorm_forward,
    llama_forcausallm_forward,
)


class LlamaDeltaZipForCausalLM(BaseDeltaZipModelForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


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


__all__ = ["LlamaDeltaZipForCausalLM"]
