# pylint: disable=unused-argument
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Set, Type
from transformers.configuration_utils import PretrainedConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)
from vllm.model_executor.parallel_utils.utils import split_tensor_along_last_dim
from .config import DeltaConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .deltazip_marlin import (
    apply_delta,
    apply_delta_embed,
    apply_delta_uncompressed,
)

ASYNC_COPY = True
logger = init_logger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class DeltaMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)

    def __str__(self):
        return f"index_mapping: {self.index_mapping}, prompt_mapping: {self.prompt_mapping}"


class BaseLayerWithDelta(nn.Module):
    def create_delta_weights(
        self, max_deltas: int, delta_config: DeltaConfig, model_config: PretrainedConfig
    ) -> None:
        """Initializes delta matrices."""
        ...

    def reset_delta(self, index: int):
        """Resets the delta weights at index back to 0."""
        ...

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        meta: torch.Tensor,
    ):
        """Overwrites delta tensors at index."""
        ...

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        """Sets the mapping indices."""
        ...


class VocabParallelEmbeddingWithDelta(BaseLayerWithDelta):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.vocab_start_index = self.base_layer.vocab_start_index
        self.vocab_end_index = self.base_layer.vocab_end_index

    def reset_delta(self, index: int):
        self.bitwidth[index] = 0
        self.delta_weights[index] = 0

    def create_delta_weights(
        self, max_deltas: int, delta_config: DeltaConfig, model_config: PretrainedConfig
    ) -> None:
        self.delta_weights = torch.zeros(
            max_deltas,
            self.base_layer.org_vocab_size // self.tp_size,
            self.base_layer.embedding_dim,
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        self.bitwidth = [0] * max_deltas

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        weight: torch.Tensor,
    ):
        self.bitwidth[index] = bitwidth
        self.delta_weights[index].copy_(weight, non_blocking=ASYNC_COPY)
        
    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.embeddings_indices = embeddings_indices
        self.indices_len = indices_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = self.indices[: self.indices_len[0]]
        if self.tp_size > 1:
            # Build the mask.
            input_mask = (x < self.vocab_start_index) | (
                x >= self.vocab_end_index)
            # Mask the input.
            masked_input = x.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = x
        output_parallel = apply_delta_embed(
            masked_input, self.delta_weights, indices, self.base_layer.weight
        )
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is VocabParallelEmbedding


class ColumnParallelLinearWithDelta(BaseLayerWithDelta):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()

    def reset_delta(self, index: int):
        self.qweight_stacked[index] = 0
        self.qzeros_stacked[index] = 0
        self.scales_stacked[index] = 0
        self.bitwidth[index] = 0

    def create_delta_weights(
        self,
        max_deltas: int,
        # let's pretend all quantization is done to the same bit width
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.qweight_stacked = torch.zeros(
            max_deltas,
            self.base_layer.weight.shape[1] // delta_config.pack_factor,
            self.base_layer.weight.shape[0],
            dtype=delta_config.delta_dtype,
            device=self.base_layer.weight.device,
        )
        self.scales_stacked = torch.zeros(
            max_deltas,
            1,
            self.base_layer.weight.shape[0],
            dtype=torch.float16,
            device=self.base_layer.weight.device,
        )
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.output_dim = self.base_layer.weight.shape[0]

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
    ):
        self.reset_delta(index)
        self.qweight_stacked[index, :, :].copy_(
            qweight, non_blocking=ASYNC_COPY)
        self.scales_stacked[index, :, :].copy_(scales, non_blocking=ASYNC_COPY)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # (note): this is not actually used.
        output = apply_delta(
            x,
            self.qweight_stacked,
            self.qzero_stacked,
            self.scales_stacked,
            self.g_idx_stacked,
            self.indices[: self.indices_len[0]],
            self.base_layer.linear_weights['weight'],
        )
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.apply_weights(x, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    @property
    def linear_weights(self):
        return self.base_layer.linear_weights

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1
        )


class MergedColumnParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.pack_factor = delta_config.pack_factor
        n_slices = 2
        self.bitwidth = [0] * max_deltas
        if not (
            len(self.base_layer.output_sizes) == n_slices
            and self.base_layer.output_sizes[0] == self.base_layer.output_sizes[1]
        ):
            raise ValueError(
                "DeltaColumnParallelLinear2Slice requires 2 slices with "
                "the same size."
            )
        self.qweight_stacked = torch.zeros(
            max_deltas,
            self.base_layer.weight.shape[1]
            // (delta_config.pack_factor * delta_config.sparse_factor * 2),
            self.base_layer.weight.shape[0] * 2,
            dtype=torch.int32,
            device=self.base_layer.weight.device,
        )
        self.scales_stacked = torch.zeros(
            max_deltas,
            1,
            self.base_layer.weight.shape[0],
            dtype=torch.float16,
            device=self.base_layer.weight.device,
        )
        self.meta_stacked = torch.zeros(
            max_deltas,
            self.base_layer.weight.shape[0],
            self.base_layer.weight.shape[1]
            // (delta_config.pack_factor * delta_config.sparse_factor),
            dtype=torch.int16,
            device=self.base_layer.weight.device,
        )
        self.indices: Optional[torch.Tensor] = None

    def reset_delta(self, index: int):
        self.bitwidth[index] = 0
        self.qweight_stacked[index] = 0
        self.scales_stacked[index] = 0
        self.meta_stacked[index] = 0

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        meta: torch.Tensor,
    ):
        self.reset_delta(index)
        self.bitwidth[index] = bitwidth
        self.qweight_stacked[index].copy_(
            qweight, non_blocking=ASYNC_COPY
        )
        self.scales_stacked[index].copy_(
            scales, non_blocking=ASYNC_COPY)
        self.meta_stacked[index].copy_(
            meta, non_blocking=ASYNC_COPY)

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        output = apply_delta(
            x,
            self.qweight_stacked,
            self.scales_stacked,
            self.meta_stacked,
            self.indices[: self.indices_len[0]],
            self.base_layer.linear_weights['weight'],
        )
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 2
        )


class MergedQKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )

        self.pack_factor = delta_config.pack_factor
        self.qweight_stacked = torch.zeros(
            max_deltas,
            self.base_layer.weight.shape[1]
            // (delta_config.pack_factor * delta_config.sparse_factor * 2),
            (self.q_proj_shard_size + self.kv_proj_shard_size * 2) * 2,
            dtype=torch.int32,
            device=self.base_layer.weight.device,
        )
        self.scales_stacked = torch.zeros(
            max_deltas,
            1,
            self.q_proj_shard_size + self.kv_proj_shard_size * 2,
            dtype=torch.float16,
            device=self.base_layer.weight.device,
        )
        self.meta_stacked = torch.zeros(
            max_deltas,
            self.q_proj_shard_size + self.kv_proj_shard_size * 2,
            self.base_layer.weight.shape[1]
            // (delta_config.pack_factor * delta_config.sparse_factor),
            dtype=torch.int16,
            device=self.base_layer.weight.device,
        )
        self.packed_indices: Optional[torch.Tensor] = None
        self.standard_indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.bitwidth = [0] * max_deltas

    def reset_delta(self, index: int):
        self.bitwidth[index] = 0
        self.qweight_stacked[index] = 0
        self.meta_stacked[index] = 0
        self.scales_stacked[index] = 0

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        meta: torch.Tensor,
    ):
        self.reset_delta(index)
        self.bitwidth[index] = bitwidth
        self.qweight_stacked[index].copy_(
            qweight, non_blocking=ASYNC_COPY
        )
        self.scales_stacked[index].copy_(scales, non_blocking=ASYNC_COPY)
        self.meta_stacked[index].copy_(meta, non_blocking=ASYNC_COPY)
        
        # print(f"set delta {index}, qweight: {qweight.shape}->{self.qweight_stacked.shape}, scales: {scales.shape}->{self.scales_stacked.shape}, meta: {meta.shape}->{self.meta_stacked.shape}")

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        output = apply_delta(
            x,
            self.qweight_stacked,
            self.scales_stacked,
            self.meta_stacked,
            self.indices[: self.indices_len[0]],
            self.base_layer.linear_weights['weight'],
        )
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is QKVParallelLinear and len(packed_modules_list) == 3


class RowParallelLinearWithDelta(BaseLayerWithDelta):
    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.bitwidth = [0] * max_deltas
        self.pack_factor = delta_config.pack_factor
        self.qweight_stacked = torch.zeros(
            (
                max_deltas,
                self.base_layer.weight.shape[1] // (
                    delta_config.pack_factor * delta_config.sparse_factor * 2),
                self.base_layer.weight.shape[0] * 2,
            ),
            dtype=delta_config.delta_dtype,
            device=self.base_layer.weight.device,
        )
        self.scales_stacked = torch.zeros(
            max_deltas,
            1,
            self.base_layer.weight.shape[0],
            dtype=torch.float16,
            device=self.base_layer.weight.device,
        )
        self.meta_stacked = torch.zeros(
            max_deltas,
            self.base_layer.weight.shape[0],
            self.base_layer.weight.shape[1] // (
                delta_config.pack_factor * delta_config.sparse_factor),
            dtype=torch.int16,
            device=self.base_layer.weight.device,
        )

    def reset_delta(self, index: int):
        self.qweight_stacked[index] = 0
        self.scales_stacked[index] = 0
        self.meta_stacked[index] = 0
        self.bitwidth[index] = 0

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        meta: torch.Tensor,
    ):
        self.reset_delta(index)
        self.bitwidth[index] = bitwidth
        self.qweight_stacked[index, :, :].copy_(
            qweight, non_blocking=ASYNC_COPY)
        self.scales_stacked[index, :, :].copy_(scales, non_blocking=ASYNC_COPY)
        self.meta_stacked[index, :, :].copy_(meta, non_blocking=ASYNC_COPY)

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_layer.bias is not None:
            raise ValueError(
                "RowParallelLinearWithDelta does not support bias yet.")
        output = apply_delta(
            x,
            self.qweight_stacked,
            self.scales_stacked,
            self.meta_stacked,
            self.indices[: self.indices_len[0]],
            self.base_layer.linear_weights['weight'],
        )
        return output

    def forward(self, input_):
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.apply_weights(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear


class LogitsProcessorWithDelta(BaseLayerWithDelta):
    def __init__(
        self,
        base_layer: LogitsProcessor,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    @property
    def logits_as_input(self):
        return self.base_layer.logits_as_input

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def scale(self):
        return self.base_layer.scale

    @property
    def org_vocab_size(self):
        return self.base_layer.org_vocab_size

    @property
    def include_gpu_probs_tensor(self):
        return self.base_layer.include_gpu_probs_tensor

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.weight_stacked = torch.zeros(
            (
                max_deltas,
                self.base_layer.vocab_size // self.tp_size,
                self.hidden_size,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.indices = None
        self.indices_padded = None
        self.indices_len = None
        self.bitwidth = [0] * max_deltas

    def reset_delta(self, index: int):
        self.weight_stacked[index] = 0
        self.bitwidth[index] = 0

    def set_delta(
        self,
        index: int,
        bitwidth: int,
        weight: torch.Tensor,
    ):
        self.reset_delta(index)
        self.bitwidth[index] = bitwidth
        self.weight_stacked[index, :, :].copy_(weight, non_blocking=ASYNC_COPY)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = sampler_indices
        self.indices_padded = sampler_indices_padded
        self.indices_len = indices_len

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        # TODO(xiaozhe): for now we assume there's no additional token added, so this simply performs additional matmuls on delta.
        logits = apply_delta_uncompressed(
            hidden_states,
            self.weight_stacked,
            self.indices[: self.indices_len[1]],
            base_embedding=embedding
        )
        logits = tensor_model_parallel_gather(logits)
        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # Special handling for the LogitsProcessor.
        return False


_all_delta_classes: Set[Type[BaseLayerWithDelta]] = {
    cls
    for cls in globals().values()
    if inspect.isclass(cls)
    and issubclass(cls, BaseLayerWithDelta)
    and cls is not BaseLayerWithDelta
}

def from_layer(
    layer: nn.Module,
    max_deltas: int,
    delta_config: DeltaConfig,
    packed_modules_list: List,
    model_config: Optional[PretrainedConfig] = None,
) -> nn.Module:
    for delta_cls in _all_delta_classes:
        if delta_cls.can_replace_layer(
            layer, delta_config, packed_modules_list, model_config
        ):
            ret = delta_cls(layer)
            ret.create_delta_weights(max_deltas, delta_config, model_config)
            return ret
    return layer

def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_deltas: int,
    delta_config: DeltaConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithDelta:
    ret = LogitsProcessorWithDelta(
        layer, lm_head.embedding_dim, lm_head.weight.dtype, lm_head.weight.device
    )
    ret.create_delta_weights(max_deltas, delta_config, model_config)
    return ret
