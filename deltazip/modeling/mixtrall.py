from ._base import *


class MixtrallDeltaZipForCausalLM(BaseDeltaZipModelForCausalLM):
  layer_type = "MixtrallBlock"
  layers_block_name = "transformer.h"
  outside_layer_modules = ["embd"]
  inside_layer_modules = ["moe.mlp.#EXPERT_ID.fc1", "moe.mlp.#EXPERT_ID.fc1"]

__all__ = ["MixtrallDeltaZipForCausalLM"]