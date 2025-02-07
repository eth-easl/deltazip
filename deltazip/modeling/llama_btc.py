from ._base import *
from ._const import EXPERT_ID_PLACEHOLDER

class LlamaBTCForCausalLM(BaseDeltaZipModelForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [f"mlp.up_proj.experts.{EXPERT_ID_PLACEHOLDER}", f"mlp.gate_proj.experts.{EXPERT_ID_PLACEHOLDER}", f"mlp.down_proj.experts.{EXPERT_ID_PLACEHOLDER}"],
    ignore_components = ["self_attn"]

__all__ = ["LlamaBTCForCausalLM"]