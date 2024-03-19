from ._base import *
from ._const import EXPERT_ID_PLACEHOLDER

class MixtrallDeltaZipForCausalLM(BaseDeltaZipModelForCausalLM):
  layer_type = "MixtrallBlock"
  layers_block_name = "transformer.h"
  outside_layer_modules = ["embd"]
  inside_layer_modules = [f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.fc1", f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.fc2"]

__all__ = ["MixtrallDeltaZipForCausalLM"]