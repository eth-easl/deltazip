import os
import re
import math
import copy
import json
import torch
import cupy as cp
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Hashable, Any, Type, Tuple
from timeit import default_timer as timer
import transformers
from transformers import AutoConfig
from vllm.logger import init_logger
from vllm.utils import LRUCache, in_wsl, total_bytes_count
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
)
from safetensors import safe_open
from .layers import (
    ModelMapping,
    from_layer,
    from_layer_logits_processor,
    BaseLayerWithPacked,
)
from .packed import ModelLayerWeights
from .config import SwapConfig
from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.model_loader import (
    _get_model_architecture,
    _set_default_torch_dtype,
)
from .utils import replace_submodule

logger = init_logger(__name__)
_GLOBAL_MODEL_ID = 0


def convert_mapping(
    mapping: ModelMapping,
    model_index_to_id: List[Optional[int]],
    max_models: int,
    vocab_size: int,
    extra_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts ModelMapping to index tensors.

    Args:
        mapping: ModelMapping mapping rows in a batch to model ids.
        delta_index_to_id: List mapping Delta ids to Delta indices.
        max_deltas: Maximum number of Deltas.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each Delta can have.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                Delta indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                Delta indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to Delta indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to Delta indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_deltas.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the Deltas, second row is for the Delta.emb
                embeddings.
            indices_len: List of lengths of the above tensors.
    """
    indices = list(mapping.index_mapping).copy()
    embedding_indices = indices.copy()
    model_indices = indices.copy()
    prompt_mapping = [
        model_index_to_id.index(x) if x > 0 else -1 for x in mapping.prompt_mapping
    ]
    swap_idx = None
    for i in range(len(indices)):
        # TODO index can be slow. optimize
        swap_idx = model_index_to_id.index(indices[i]) if indices[i] > 0 else -1
        embedding_indices[i] = swap_idx if indices[i] > 0 else 0
        indices[i] = i
        model_indices[i] = swap_idx

    indices = torch.tensor(
        [indices, model_indices, embedding_indices], dtype=torch.long, device="cuda"
    )
    prompt_mapping = torch.tensor(prompt_mapping, device="cuda", dtype=torch.long)
    embeddings_indices = torch.stack(
        [indices[2] * extra_vocab_size, indices[2] * (vocab_size + extra_vocab_size)]
    )
    embeddings_indices[embeddings_indices == -1] = max_models - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_models - 1
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), device="cuda", dtype=torch.long
    ) + (sampler_indices_padded * len(sampler_indices_padded))

    indices_len = (
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1],
        embeddings_indices.shape[-1],
    )

    return (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        indices_len,
    )


def get_model_id():
    global _GLOBAL_MODEL_ID
    _GLOBAL_MODEL_ID += 1
    return _GLOBAL_MODEL_ID


class SwapModel:
    def __init__(self, swap_model_id: int, swaps: Dict[str, ModelLayerWeights]):
        self.id = swap_model_id
        self.swaps: Dict[str, ModelLayerWeights] = swaps

    def get_swap(self, module_name: str) -> Optional[ModelLayerWeights]:
        return self.swaps.get(module_name, None)

    @classmethod
    def from_checkpoint(
        cls,
        path_or_name,
        id,
        model_config: ModelConfig,
        device: torch.device,
        trust_remote_code: bool = False,
    ):
        start = timer()
        model_class = _get_model_architecture(model_config)
        with _set_default_torch_dtype(model_config.dtype):
            with torch.device("cpu"):
                model = model_class(model_config.hf_config)
                model.load_weights(
                    path_or_name,
                    model_config.download_dir,
                    model_config.load_format,
                    model_config.revision,
                )
        model = model.eval()
        modules = {}

        for module_name, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue
            modules[module_name] = ModelLayerWeights(
                module_name=module_name,
                weight=module.weight,
                bias=module.bias if hasattr(module, "bias") else None,
            )
        end = timer()
        logger.info(f"Disk -> CPU: Loaded in {end - start:.3f} seconds")
        return cls(id, modules)


class SwapModelManager:
    """A manager that manages multiple SwapModels."""

    def __init__(
        self,
        model,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        swap_config: SwapConfig,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.max_num_seqs = max_num_seqs
        self.swap_config = swap_config
        assert (
            self.capacity >= self.packed_swap_slots
        ), "Capacity must be greater than packed swap slots"
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.swap_index_to_id: List[Optional[int]] = [None] * self.packed_swap_slots
        self.vocab_size = vocab_size
        self.base_indices = torch.empty(
            self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.sampler_indices = torch.empty(
            self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.sampler_indices_padded = torch.empty(
            self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.embeddings_indices = torch.empty(
            2, self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.offset = []
        self.indices_len = []
        self.model = model
        if hasattr(self.model, "supported_swap_modules"):
            self.supported_swap_modules = copy.deepcopy(
                self.model.supported_swap_modules
            )

            self.packed_modules_mapping = copy.deepcopy(
                self.model.packed_modules_mapping
            )
        self.packed_modules: Dict[str, List[str]] = {}
        self.modules: Dict[str, "ModelLayerWeights"] = {}

        self._registered_swaps: Dict[str, "SwapModel"] = {}
        self._active_swaps: Dict[int, None] = {}

        self._last_mapping = None
        self._create_swap_modules()
        self.model.swap_manager = self

    @property
    def packed_swap_slots(self) -> int:
        return self.swap_config.max_packed_model

    @property
    def capacity(self) -> int:
        return self.swap_config.max_cpu_model

    def __len__(self) -> int:
        return len(self._registered_swaps)

    def activate_swap(self, swap_id: int):
        if swap_id in self._active_swaps:
            return False
        first_free_slot = next(
            (
                (i, swap_id)
                for i, swap_id in enumerate(self.swap_index_to_id)
                if swap_id is None
            ),
            None,
        )
        if first_free_slot is None:
            raise ValueError("No free swap slots")

        index, _ = first_free_slot
        self._active_swaps[swap_id] = None
        swap_model = self._registered_swaps[swap_id]
        self.swap_index_to_id[index] = swap_model.id

        for module_name, module in self.modules.items():
            module_swap = swap_model.get_swap(module_name)
            if module_swap:
                module.set_pack(index, module_swap.weight)
            else:
                module.reset_pack(index)
        return True

    def clear_base_module(self):
        for module_name, module in self.modules.items():
            module.clear_base()

    def _deactivate_swap(self, swap_id: int):
        try:
            index = self.swap_index_to_id.index(swap_id)
            self.swap_index_to_id[index] = None
        except ValueError:
            pass

    def deactivate_swap(self, swap_id: int) -> bool:
        if swap_id in self._active_swaps:
            self._deactivate_swap(swap_id)
            self._active_swaps.pop(swap_id)
            return True
        return False

    def _add_swap(self, swap: SwapModel) -> bool:
        self._create_merged_swap_inplace(swap)
        self._registered_swaps[swap.id] = swap

    def add_swap(self, swap: SwapModel) -> bool:
        """Add a SwapModel to the manager CPU cache."""
        if swap.id not in self._registered_swaps:
            if len(self._registered_swaps) >= self.capacity:
                raise RuntimeError("No free swaps slots.")
            self._add_swap(swap)
            return True
        return False

    def remove_swap(self, swap_id: int) -> bool:
        """Remove a SwapModel from the manager CPU cache."""
        self.deactivate_swap(swap_id)
        return bool(self._registered_swaps.pop(swap_id, None))

    # TODO see if this can be vectorized
    def _set_swap_mapping(self, mapping: ModelMapping) -> None:
        (
            base_indices,
            sampler_indices,
            sampler_indices_padded,
            embeddings_indices,
            indices_len,
        ) = convert_mapping(
            mapping,
            self.swap_index_to_id,
            self.packed_swap_slots + 1,
            self.vocab_size,
            0,
        )
        self.base_indices[: base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[: sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[: sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded
        )
        self.embeddings_indices[
            : embeddings_indices.shape[0], : embeddings_indices.shape[1]
        ].copy_(embeddings_indices)
        # Maintain the reference
        self.indices_len[:] = indices_len

    def set_swap_mapping(self, swap_mapping: ModelMapping) -> None:
        if self._last_mapping != swap_mapping:
            self._set_swap_mapping(swap_mapping)
        self._last_mapping = swap_mapping

    def list_swaps(self) -> Dict[int, SwapModel]:
        """List all registered SwapModels."""
        return dict(self._registered_swaps)

    def get_swap(self, swap_id: int) -> Optional[SwapModel]:
        return self._registered_swaps.get(swap_id, None)

    def remove_all_swaps(self) -> bool:
        """Remove all SwapModels from the manager."""
        self._registered_swaps.clear()
        self.swap_index_to_id = [None] * self.packed_swap_slots
        self._active_swaps.clear()

    def _create_swap_modules(self):
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name):
                continue
            parts = module_name.split(".")[-1]
            packed_moduled_list = self.packed_modules_mapping.get(parts, [])
            new_module = replace_submodule(
                self.model,
                module_name,
                from_layer(
                    module,
                    self.packed_swap_slots,
                    self.swap_config,
                    packed_moduled_list,
                    self.model.config,
                ),
            )
            if "lm_head" in module_name:
                logits_processor_module = self.model.get_submodule("logits_processor")
                new_module = replace_submodule(
                    self.model,
                    "logits_processor",
                    from_layer_logits_processor(
                        logits_processor_module,
                        module,
                        self.packed_swap_slots,
                        self.swap_config,
                        self.model.config,
                    ),
                )
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            new_module.set_mapping(
                self.base_indices,
                self.sampler_indices,
                self.sampler_indices_padded,
                self.embeddings_indices,
                self.indices_len,
            )

    def register_module(self, module_name: str, module: "BaseLayerWithPacked"):
        assert isinstance(module, BaseLayerWithPacked)
        self.modules[module_name] = module

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module), module_name
            )
            or target_module == module_name
            for target_module in self.supported_swap_modules
        )

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name)
        if not replacements:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_swap_inplace(self, swap_model: SwapModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_swaps = []
            has_replacement = False
            for r in new_module_names:
                model = swap_model.get_swap(r)
                replacement_swaps.append(model)
                if model:
                    has_replacement = True
            if not has_replacement:
                continue
            for i in range(len(replacement_swaps)):
                if replacement_swaps[i]:
                    continue
                replacement_swaps[i] = None
            swap_model.swaps[module_name] = ModelLayerWeights.pack(replacement_swaps)


class SwapLRUCache(LRUCache):
    def __init__(self, capacity: int, deactivate_swap_fn: Callable[[Hashable], None]):
        super().__init__(capacity)
        self.deactivate_swap_fn = deactivate_swap_fn

    def _on_remove(self, key: Hashable, value: Any):
        self.deactivate_swap_fn(key)
        return super()._on_remove(key, value)


class LRUCacheSwapModelManager(SwapModelManager):
    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        swap_config: SwapConfig,
        model_config: ModelConfig,
    ):
        super().__init__(
            model,
            max_num_seqs,
            max_num_batched_tokens,
            vocab_size,
            swap_config,
            model_config=model_config,
        )
        self._registered_swaps: SwapLRUCache = SwapLRUCache(
            self.capacity, self.deactivate_swap
        )
        self._active_swaps: SwapLRUCache = SwapLRUCache(
            self.packed_swap_slots, self._deactivate_swap
        )

    def list_swaps(self) -> Dict[int, SwapModel]:
        """List all registered SwapModels."""
        return dict(self._registered_swaps.cache)

    def add_swap(self, swap: SwapModel) -> bool:
        """Add a SwapModel to the manager."""
        if swap.id not in self._registered_swaps:
            self._add_swap(swap)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_swaps.touch(swap.id)
            was_added = False
        return was_added

    def activate_swap(
        self,
        swap_id: int,
    ) -> bool:
        if (
            swap_id not in self._active_swaps
            and len(self._active_swaps) >= self.packed_swap_slots
        ):
            self._active_swaps.remove_oldest()
        result = super().activate_swap(swap_id)
        # We always touch to update the LRU cache order
        self._active_swaps.touch(swap_id)
        return result

    def remove_oldest_swap(self) -> bool:
        if len(self._registered_swaps) > 0:
            self._registered_swaps.remove_oldest()
            return True
        return False


def create_swap_manager(
    model: nn.Module,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    vocab_size: int,
    swap_config: SwapConfig,
    model_config: ModelConfig,
    swap_manager_cls: Type[SwapModelManager] = SwapModelManager,
    **kwargs,
) -> SwapModelManager:
    if not hasattr(model, "supported_swap_modules"):
        raise ValueError(f"Model {type(model)} is not supported for Swap.")
    swap_manager = swap_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        swap_config=swap_config,
        model_config=model_config,
        **kwargs,
    )
    return swap_manager
