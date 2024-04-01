from ._base import *
from ._const import EXPERT_ID_PLACEHOLDER

class MixtrallDeltaZipForCausalLM(BaseDeltaZipModelForCausalLM):
  layer_type = "MixtrallBlock"
  layers_block_name = "transformer.h"
  outside_layer_modules = ["transformer.embd", "lm_head", ]
  inside_layer_modules = [f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.fc1", f"moe.mlp.{EXPERT_ID_PLACEHOLDER}.fc2"]
  ignore_components = ["mixer", "gate"]

__all__ = ["MixtrallDeltaZipForCausalLM"]