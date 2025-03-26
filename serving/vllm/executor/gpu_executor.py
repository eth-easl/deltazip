from typing import Dict, List, Optional

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VisionLanguageConfig,
)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata, SequenceGroup
from vllm.utils import get_distributed_init_method, get_ip, get_open_port, make_async
from vllm.delta.config import DeltaConfig
from vllm.delta.request import DeltaRequest
from vllm.swap.config import SwapConfig
from vllm.swap.request import SwapRequest

logger = init_logger(__name__)


class GPUExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        delta_config: Optional[DeltaConfig],
        swap_config: Optional[SwapConfig],
        vision_language_config: Optional[VisionLanguageConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.delta_config = delta_config
        self.swap_config = swap_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.vision_language_config = vision_language_config

        # Instantiate the worker and load the model to GPU.
        self._init_worker()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

    def _init_worker(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        assert (
            self.parallel_config.world_size == 1
        ), "GPUExecutor only supports single GPU."

        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            delta_config=self.delta_config,
            swap_config=self.swap_config,
            vision_language_config=self.vision_language_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def reload_model(self, model_path_or_name: str) -> None:
        self.driver_worker.reload_model_weights(model_path_or_name)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine first profiles the existing memory usage.
        Then, it allocates the remaining memory for KV blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        (
            num_gpu_blocks,
            num_cpu_blocks,
        ) = self.driver_worker.profile_num_available_blocks(
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_dtype=self.cache_config.cache_dtype,
        )

        logger.info(
            f"# GPU blocks: {num_gpu_blocks}, " f"# CPU blocks: {num_cpu_blocks}"
        )

        check_block_size_valid(
            num_gpu_blocks,
            self.cache_config.block_size,
            self.model_config.max_model_len,
        )

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self.driver_worker.init_cache_engine(cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self.driver_worker.warm_up_model()

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        sequence_groups: List[SequenceGroup],
    ) -> SamplerOutput:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            sequence_groups=sequence_groups,
        )
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> List[int]:
        return self.driver_worker.list_loras()

    def add_delta(self, delta_request: DeltaRequest) -> bool:
        assert delta_request.delta_int_id > 0, "delta_id must be greater than 0."
        return self.driver_worker.add_delta(delta_request)

    def remove_delta(self, delta_id: int) -> bool:
        assert delta_id > 0, "delta_id must be greater than 0."
        return self.driver_worker.remove_delta(delta_id)

    def list_deltas(self) -> List[int]:
        return self.driver_worker.list_deltas()

    def add_swap(self, swap_request: SwapRequest) -> bool:
        assert swap_request.swap_int_id > 0, "swap_id must be greater than 0."
        return self.driver_worker.add_swap(swap_request)

    def remove_delta(self, swap_id: int) -> bool:
        assert swap_id > 0, "swap_id must be greater than 0."
        return self.driver_worker.remove_swap(swap_id)

    def list_swaps(self) -> List[int]:
        return self.driver_worker.list_swaps()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return

    def prefetch_deltas(self, delta_request: DeltaRequest) -> bool:
        assert delta_request.delta_int_id > 0, "delta_id must be greater than 0."
        return self.driver_worker.prefetch_deltas(delta_request)


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        sequence_groups: List[SequenceGroup],
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        return output

    async def check_health_async(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return
