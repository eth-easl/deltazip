from ._base import *
from ._const import EXPERT_ID_PLACEHOLDER

class GPTNeoXMoeDeltaZipForCausalLM(BaseDeltaZipModelForCausalLM):
    layer_type = "GPTNeoXLayer"
    layers_block_name = "gpt_neox.layers"
    outside_layer_modules = ["gpt_neox.embed_in", "gpt_neox.final_layer_norm"]
    inside_layer_modules = [f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.dense_h_to_4h", f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.dense_4h_to_h"]
    ignore_components = ["gate"]

__all__ = ["GPTNeoXMoeDeltaZipForCausalLM"]