import torch
import cupy as cp
from typing import Dict
from src.lossless.nvcomp import LZ4Manager
from src.lossless.nvcomp import SnappyManager
from src.lossless.nvcomp import BitcompManager
from src.lossless.nvcomp import GdeflateManager
from src.lossless.nvcomp import CascadedManager
from torch.utils.dlpack import to_dlpack, from_dlpack

dtype_maps = {
    'fp16': torch.float16,
    'fp32': torch.float32,
}
cp_dtype_maps = {
    'fp16': cp.float16,
    'fp32': cp.float32,
}
bytes_nums = {
    'fp16': 2,
    'fp32': 4,
}

class LosslessCompressor():
    def __init__(self, algorithm: str, dtype='fp16') -> None:
        self._dtype = dtype
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
        self.comp_manager.input_type = cp_dtype_maps[dtype]
    
    def compress_tensor(self, tensor: torch.Tensor):
        tensor.requires_grad_(False)
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        # zero-copy to cupy format
        to_compress_tensor = cp.from_dlpack(to_dlpack(tensor))
        tensor_shape = to_compress_tensor.shape
        compressed_tensor = self.comp_manager.compress(to_compress_tensor)
        del to_compress_tensor
        torch.cuda.empty_cache()
        return cp.asnumpy(compressed_tensor), tensor_shape

    def decompress_tensor(self, compressed_tensor: cp.ndarray, tensor_shape: tuple):
        decompressed_tensor = self.comp_manager.decompress(compressed_tensor)
        torch_tensor = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), tensor_shape)
        del decompressed_tensor
        torch.cuda.empty_cache()
        return torch_tensor

    def compress_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        tensors = {}
        tensors_shape = {}
        for key in state_dict:
            tensors[key], tensors_shape[key] = self.compress_tensor(state_dict[key])
        return tensors, tensors_shape

    def decompress_state_dict(self, compressed_state_dict: Dict[str, torch.Tensor], tensor_shapes: Dict[str, tuple]):
        tensors = {}
        for key in compressed_state_dict:
            tensors[key] = self.decompress_tensor(compressed_state_dict[key], tensor_shapes[key])
        return tensors