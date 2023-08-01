import torch
import cupy as cp
from typing import Dict
from fmzip.lossless.nvcomp import LZ4Manager
from fmzip.lossless.nvcomp import SnappyManager
from fmzip.lossless.nvcomp import BitcompManager
from fmzip.lossless.nvcomp import GdeflateManager
from fmzip.lossless.nvcomp import CascadedManager
from loguru import logger
from torch.utils.dlpack import to_dlpack, from_dlpack

dtype_maps = {
    'int8': torch.int8,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'int32': torch.int32
}

cp_dtype_maps = {
    'int8': cp.int8,
    'fp16': cp.float16,
    'fp32': cp.float32,
    'int32': cp.int32
}

class LosslessCompressor():
    def __init__(self, algorithm: str='gdeflate') -> None:
        if algorithm == 'gdeflate':
            self.comp_manager = GdeflateManager()
        elif algorithm == 'lz4':
            self.comp_manager = LZ4Manager()
        elif algorithm == 'snappy':
            self.comp_manager = SnappyManager()
        elif algorithm == 'bitcomp':
            self.comp_manager = BitcompManager()
        elif algorithm == 'cascaded':
            self.comp_manager = CascadedManager()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm},  supported algorithms: ['gdeflate', 'lz4', 'snappy', 'bitcomp', 'cascaded']")
    
    def compress_tensor(self, tensor: torch.Tensor):
        tensor.requires_grad_(False)
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        # zero-copy to cupy format
        tensor_shape = tensor.shape
        to_compress_tensor = cp.from_dlpack(to_dlpack(tensor))
        logger.debug(f"compressiong dtype {tensor.dtype}")
        if tensor.dtype == torch.int8:
            dtype = 'int8'
            self.comp_manager.input_type = cp.int8
        elif tensor.dtype == torch.float16:
            dtype = 'fp16'
            self.comp_manager.input_type = cp.float16
        elif tensor.dtype == torch.int32:
            dtype = 'int32'
            self.comp_manager.input_type = cp.int32
        else:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        compressed_tensor = self.comp_manager.compress(to_compress_tensor)
        return cp.asnumpy(compressed_tensor), tensor_shape, dtype

    def decompress_tensor(self, compressed_tensor: cp.array, tensor_shape: tuple, dtype='fp16'):
        self.comp_manager.input_type = cp_dtype_maps[dtype]
        decompressed_tensor = self.comp_manager.decompress(compressed_tensor)
        torch_tensor = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), tensor_shape)
        return torch_tensor

    def compress_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        tensors = {}
        tensors_shape = {}
        tensors_dtype = {}
        for key in state_dict:
            logger.debug(f"compressiong {key}, shape: {state_dict[key].shape}, dtype: {state_dict[key].dtype}")
            tensors[key], tensors_shape[key], tensors_dtype[key] = self.compress_tensor(state_dict[key])
        return tensors, tensors_shape, tensors_dtype

    def decompress_state_dict(
            self, 
            compressed_state_dict: Dict[str, cp.array], 
            tensor_shapes: Dict[str, tuple], 
            tensor_dtypes: Dict[str, str]=None
        ):
        tensors = {}
        for key in compressed_state_dict.keys():
            tensors[key] = self.decompress_tensor(compressed_state_dict[key], tensor_shapes[key], tensor_dtypes[key])
        return tensors