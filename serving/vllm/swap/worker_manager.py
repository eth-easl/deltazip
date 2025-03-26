from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import Any, List, Optional, Set, Type, Dict
import torch
import time
from .layers import ModelMapping
from .request import SwapRequest
from .config import SwapConfig
from vllm.logger import init_logger
from .models import (
    SwapModel,
    SwapModelManager,
    LRUCacheSwapModelManager,
    create_swap_manager,
)
from vllm.config import ModelConfig
from vllm.sequence import SequenceGroup

logger = init_logger(__name__)
LOG_TIME = False


class AbstractWorkerManager(ABC):
    """Abstract class for managing LoRA/Delta/Swap models on the worker side."""

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        model_config: ModelConfig,
        swap_config: SwapConfig,
        device: torch.device,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.model_config = model_config
        self.swap_config = swap_config

    @property
    @abstractmethod
    def is_enabled(self) -> bool: ...

    @abstractmethod
    def create_swap_manager(
        self,
        model: torch.nn.Module,
    ) -> Any: ...

    @abstractmethod
    def set_active_swaps(
        self, swap_requests: List[SwapRequest], swap_mapping: ModelMapping
    ) -> None: ...

    @abstractmethod
    def add_swap(self, swap_request: SwapRequest) -> bool: ...

    @abstractmethod
    def add_dummy_swap(self, swap_request: SwapRequest) -> bool: ...

    @abstractmethod
    def remove_swap(self, swap_id: int) -> bool: ...

    @abstractmethod
    def remove_all_swaps(self) -> bool: ...

    @abstractmethod
    def list_swaps(self) -> Set[int]: ...


class WorkerSwapManager(AbstractWorkerManager):
    """WorkerSwapManager manages the swaps on the worker side."""

    _swap_manager_cls: Type

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        model_config: ModelConfig,
        swap_config: SwapConfig,
        device: torch.device,
        embedding_modules: Dict[str, str],
        embedding_padding_modules: List[str],
        swap_model_cls: Type[SwapModel] = SwapModel,
    ):
        self._swap_manager: Optional[SwapModelManager] = None
        self._swap_model_cls = swap_model_cls
        self.embedding_modules = embedding_modules
        self.embedding_padding_modules = embedding_padding_modules
        super().__init__(
            max_num_seqs,
            max_num_batched_tokens,
            vocab_size,
            model_config,
            swap_config,
            device,
        )

    @property
    def is_enabled(self) -> bool:
        return True

    def create_swap_manager(self, model: torch.nn.Module) -> Any:
        swap_manager = create_swap_manager(
            model,
            swap_manager_cls=self._swap_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            swap_config=self.swap_config,
            model_config=self.model_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._swap_manager: SwapModelManager = swap_manager
        return swap_manager.model

    def set_active_swaps(
        self,
        swap_requests: List[SwapRequest],
        swap_mapping: ModelMapping,
        sequence_groups: List[SequenceGroup],
    ) -> None:
        self._apply_swaps(swap_requests, sequence_groups)
        self._swap_manager.set_swap_mapping(swap_mapping)

    def _apply_swaps(
        self, swap_requests: List[SwapRequest], sequence_groups: List[SequenceGroup]
    ) -> None:
        swaps_that_exist = self.list_swaps()
        swaps_map = {
            swap_request.swap_int_id: swap_request for swap_request in swap_requests
        }
        if len(swaps_map) > self._swap_manager.packed_swap_slots:
            raise RuntimeError(
                f"Number of requested deltas ({len(swaps_map)}) is greater than the number of GPU delta slots "
                f"({self._swap_manager.packed_swap_slots})."
            )
        new_swaps = set(swaps_map)
        swaps_to_add = new_swaps - swaps_that_exist
        swaps_to_remove = swaps_that_exist - new_swaps

        for swap_id in swaps_to_remove:
            self.remove_swap(swap_id)

        for swap_id in swaps_to_add:
            self.add_swap(swaps_map[swap_id], sequence_groups)

    def _load_swap(self, swap_request: SwapRequest) -> SwapModel:
        try:
            # TODO(xiaozhe): actual loading logic here
            swap = self._swap_model_cls.from_checkpoint(
                swap_request.swap_local_path,
                id=swap_request.swap_int_id,
                model_config=self.model_config,
                device=self.device,
            )
        except Exception as e:
            logger.error(
                f"Failed to load swap model from {swap_request.swap_local_path}: {e}"
            )
            return None
        return swap

    def add_dummy_swap(self, swap_request: SwapRequest) -> bool:
        if swap_request.swap_int_id in self.list_swaps():
            return False
        raise NotImplementedError

    def add_swap(
        self, swap_request: SwapRequest, sequence_groups: List[SequenceGroup]
    ) -> bool:
        if swap_request.swap_int_id in self.list_swaps():
            return False
        swap = self._load_swap(swap_request)
        for sg in sequence_groups:
            sg.maybe_set_cpu_loading_time(time.time())
        loaded = self._swap_manager.add_swap(swap)
        self._swap_manager.activate_swap(swap.id)
        return loaded

    def remove_swap(self, swap_id: int) -> bool:
        return self._swap_manager.remove_swap(swap_id)

    def remove_all_swaps(self) -> bool:
        return self._swap_manager.remove_all_swaps()

    def list_swaps(self) -> Set[int]:
        return set(self._swap_manager.list_swaps())


class LRUCacheWorkerSwapManager(WorkerSwapManager):
    _swap_manager_cls = LRUCacheSwapModelManager

    def create_swap_manager(self, model) -> Any:
        swap_manager = create_swap_manager(
            model,
            swap_manager_cls=self._swap_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            swap_config=self.swap_config,
            model_config=self.model_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._swap_manager: LRUCacheSwapModelManager = swap_manager
        return swap_manager.model

    def _apply_swaps(
        self, swap_requests: List[SwapRequest], sequence_groups: List[SequenceGroup]
    ) -> None:
        swap_maps = {
            swap_request.swap_int_id: swap_request for swap_request in swap_requests
        }
        if len(swap_maps) > self._swap_manager.packed_swap_slots:
            raise RuntimeError(
                f"Number of requested swap ({len(swap_maps)}) is greater than the number of GPU swap slots "
                f"({self._swap_manager.packed_swap_slots})."
            )
        for swap in swap_maps.values():
            self.add_swap(swap, sequence_groups)

    def clear_base(self):
        self._swap_manager.clear_base_module()

    def add_swap(
        self, swap_request: SwapRequest, sequence_groups: List[SequenceGroup]
    ) -> bool:
        if swap_request.swap_int_id not in self.list_swaps():
            if len(self._swap_manager) + 1 > self._swap_manager.capacity:
                self._swap_manager.remove_oldest_swap()
            swap = self._load_swap(swap_request)
            loaded = self._swap_manager.add_swap(swap)
        else:
            loaded = self._swap_manager.get_swap(swap_request.swap_int_id)
        for sg in sequence_groups:
            sg.maybe_set_cpu_loading_time(time.time())
        self._activate_swap(swap_request=swap_request)
        for sg in sequence_groups:
            sg.maybe_set_gpu_loading_time(time.time())
        return loaded

    def _activate_swap(self, swap_request: SwapRequest):
        global LOG_TIME
        start = timer()
        self._swap_manager.activate_swap(swap_request.swap_int_id)
        end = timer()
        if not LOG_TIME:
            logger.info(f"[{time.time()}] CPU -> GPU time: {end - start:.4f}")
            LOG_TIME = True
