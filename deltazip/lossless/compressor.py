import torch
import cupy as cp
from typing import Dict
from deltazip.lossless.nvcomp import LZ4Manager
from deltazip.lossless.nvcomp import SnappyManager
from deltazip.lossless.nvcomp import BitcompManager
from deltazip.lossless.nvcomp import GdeflateManager
from deltazip.lossless.nvcomp import CascadedManager
from torch.utils.dlpack import to_dlpack, from_dlpack
from loguru import logger

dtype_maps = {
    "int8": torch.int8,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "bool": torch.bool,
}

cp_dtype_maps = {
    "int8": cp.int8,
    "fp16": cp.float16,
    "fp32": cp.float32,
    "int32": cp.int32,
    "bool": cp.bool_
}


class LosslessCompressor:
    def __init__(self, algorithm: str = "gdeflate", device_id: int = 0) -> None:
        if algorithm == "gdeflate":
            self.comp_manager = GdeflateManager(device_id=device_id)
        elif algorithm == "lz4":
            self.comp_manager = LZ4Manager(device_id=device_id)
        elif algorithm == "snappy":
            self.comp_manager = SnappyManager(device_id=device_id)
        elif algorithm == "bitcomp":
            self.comp_manager = BitcompManager(device_id=device_id)
        elif algorithm == "cascaded":
            self.comp_manager = CascadedManager(device_id=device_id)
        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm},  supported algorithms: ['gdeflate', 'lz4', 'snappy', 'bitcomp', 'cascaded']"
            )

    def compress_tensor(self, tensor: torch.Tensor):
        tensor.requires_grad_(False)
        tensor_shape = tensor.shape
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        to_compress_tensor = cp.from_dlpack(to_dlpack(tensor))
        # logger.debug(f"compressiong dtype {tensor.dtype}")
        if tensor.dtype == torch.int8:
            dtype = "int8"
            self.comp_manager.input_type = cp.int8
        elif tensor.dtype == torch.float16:
            dtype = "fp16"
            self.comp_manager.input_type = cp.float16
        elif tensor.dtype == torch.int32:
            dtype = "int32"
            self.comp_manager.input_type = cp.int32
        elif tensor.dtype == torch.float32:
            dtype = "fp32"
            self.comp_manager.input_type = cp.float32
        elif tensor.dtype == torch.bool:
            dtype = "bool"
            self.comp_manager.input_type = cp.bool_
        else:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        compressed_tensor = self.comp_manager.compress(to_compress_tensor)
        return cp.asnumpy(compressed_tensor), tensor_shape, dtype

    def decompress_tensor(
        self,
        compressed_tensor: cp.array,
        tensor_shape: tuple,
        dtype="fp16",
        target_device="cuda:0",
    ):
        self.comp_manager.input_type = cp_dtype_maps[dtype]
        decompressed_tensor = self.comp_manager.decompress(compressed_tensor)
        torch_tensor = torch.reshape(
            from_dlpack(decompressed_tensor.toDlpack()), tensor_shape
        )
        return torch_tensor.to(torch.device(target_device))

    def compress_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        tensors = {}
        tensors_shape = {}
        tensors_dtype = {}
        for key in state_dict:
            print(key)
            tensors[key], tensors_shape[key], tensors_dtype[key] = self.compress_tensor(
                state_dict[key]
            )
        return tensors, tensors_shape, tensors_dtype

    def decompress_state_dict(
        self,
        compressed_state_dict: Dict[str, cp.array],
        tensor_shapes: Dict[str, tuple],
        tensor_dtypes: Dict[str, str] = None,
        use_bfloat16: bool = False,
        target_device: str = "cuda:0",
    ):
        with torch.no_grad():
            tensors = {}
            for key in compressed_state_dict.keys():
                decompressed = self.decompress_tensor(
                    compressed_state_dict[key],
                    tensor_shapes[key],
                    tensor_dtypes[key],
                    target_device,
                )
                if use_bfloat16:
                    tensors[key] = decompressed.bfloat16()
                else:
                    tensors[key] = decompressed
            return tensors
