import os
import json
import torch
import cupy as cp
from typing import Tuple
from loguru import logger
from safetensors.numpy import save_file
from torch.utils.dlpack import to_dlpack
from timeit import default_timer as timer
from torch.utils.dlpack import from_dlpack
from transformers import AutoModelForCausalLM
from src.lossless.nvcomp import GdeflateManager as manager

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
class CompressedInferenceService():
    def __init__(self, base_model: str, dtype='fp16') -> None:
        self._dtype = dtype
        self.dtype = dtype_maps[dtype]
        self._init_base_model(base_model)
        self.comp_manager = manager()
        self.comp_manager.input_type = cp_dtype_maps[dtype]

    def _init_base_model(self, base_model: str):
        logger.debug("Loading base model: {}".format(base_model))
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=self.dtype)
        self.base_model.cuda()
        self.base_model.requires_grad_(False)
        logger.debug("Done loading base model")

    def compress_delta_model(self, target_model: str, dest: str, low_gpu_mem=True)-> Tuple[float, float]:
        logger.debug("Loading target model: {}".format(target_model))
        target_model = AutoModelForCausalLM.from_pretrained(target_model, torch_dtype=self.dtype)
        target_model.cuda()
        target_model.requires_grad_(False)
        with torch.no_grad():
            total_params = sum(p.numel() for p in target_model.parameters())
            total_bytes = bytes_nums[self._dtype] * total_params
            for name, param in target_model.named_parameters():
                param -= self.base_model.state_dict()[name]
        # now target_model is a delta model
        if low_gpu_mem:
            self.base_model = self.base_model.cpu()
            torch.cuda.empty_cache()
        target_model.requires_grad_(False)
        logger.debug("Done loading target model")
        tensor_shapes = {}
        tensors = {}
        timer_start = timer()
        for name, param in target_model.named_parameters():
            to_compress_tensor = cp.from_dlpack(to_dlpack(param))
            tensor_shape = to_compress_tensor.shape
            compressed_tensor = self.comp_manager.compress(to_compress_tensor)
            tensor_shapes[name] = list(tensor_shape)
            tensors[name] = cp.asnumpy(compressed_tensor)
        timer_end = timer()
        with open(os.path.join(dest, "tensor_shapes.json"), "w") as fp:
            json.dump(tensor_shapes, fp)
        save_file(tensors, os.path.join(dest, "compressed_model.safetensors"))
        # read the compressed file and calculate the size in bytes
        compressed_size = os.path.getsize(os.path.join(dest, "compressed_model.safetensors"))
        del tensors
        del tensor_shapes
        torch.cuda.empty_cache()
        return total_bytes / compressed_size, timer_end - timer_start
    
    def register_service(self, src_directory: str, low_gpu_mem=True):
        pass

    def decompress_delta_model():
        pass
