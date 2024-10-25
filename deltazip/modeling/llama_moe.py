from ._base import *
from ._const import EXPERT_ID_PLACEHOLDER

class LlamaMoeDeltaZipForCausalLM(BaseDeltaZipModelForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.up_proj", f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.gate_proj", f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.down_proj"],
    ignore_components = ["gate"]

__all__ = ["LlamaMoeDeltaZipForCausalLM"]