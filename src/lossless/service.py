import os
import json
import torch
import cupy as cp
from typing import Tuple
from loguru import logger
from safetensors import safe_open
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
        self.services = {}
        self.services[base_model] = {
            'dest': 'gpu_memory',
            'model': self.base_model,
            'hit': 0,
        }
        self.layer_meta = {}
        self.comp_manager = manager()
        self.comp_manager.input_type = cp_dtype_maps[dtype]

    def _init_base_model(self, base_model: str):
        logger.debug("Loading base model: {}".format(base_model))
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=self.dtype)
        self.base_model.cuda()
        self.base_model.requires_grad_(False)
        logger.debug("Done loading base model")

    def compress_model(self, target_model: str, dest: str, low_gpu_mem=True, delta=True)-> Tuple[float, float]:
        logger.debug("Loading target model: {}".format(target_model))
        target_model = AutoModelForCausalLM.from_pretrained(target_model, torch_dtype=self.dtype)
        if low_gpu_mem:
            self.base_model = self.base_model.cpu()
            torch.cuda.empty_cache()
        else:
            target_model.cuda()
        target_model.requires_grad_(False)
        total_params = sum(p.numel() for p in target_model.parameters())
        total_bytes = bytes_nums[self._dtype] * total_params
        if delta:
            with torch.no_grad():
                for name, param in target_model.named_parameters():
                    param -= self.base_model.state_dict()[name]
        # now target_model is a delta model
        target_model.requires_grad_(False)
        target_model.cuda()
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
    
    def register_service(self, src_directory: str, dest: str, low_gpu_mem=True, delta=True) -> float:
        timer_start = timer()
        assert dest in ['disk', 'host_memory', 'gpu_memory'], "dest must be either disk, host_memory or gpu_memory"
        assert os.path.exists(src_directory), "src_directory must exist"
        if dest == 'disk':
            # do nothing
            dest = os.path.join(src_directory, "compressed_model.safetensors")
            self.services[src_directory] = {
                'dest': dest,
                'model': None,
                'hit': 0,
            }
        elif dest == 'host_memory':
            with safe_open(os.path.join(src_directory, "compressed_model.safetensors"), framework='np', device="cpu") as f:
                tensors = {}
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
                self.services[src_directory] = {
                    'dest': dest,
                    'model': tensors,
                    'hit': 0,
                }
        elif dest == 'gpu_memory':
            tensors = {}
            for key in f.keys():
                tensors[key] = cp.array(f.get_tensor(key))
            self.services[src_directory] = {
                'dest': dest,
                'model': tensors,
                'hit': 0,
            }
        with open(os.path.join(src_directory, "tensor_shapes.json"), "r") as fp:
            self.layer_meta = json.load(fp)
        timer_end = timer()
        return timer_end - timer_start
    
    def restore_model(self, src_directory: str, target_model: str) -> float:
        if src_directory not in self.services:
            logger.warning(f"src_directory {src_directory} not registered, registering now. When possible, please register the service before calling restore_model")
            self.register_service(src_directory, 'gpu_memory')
        if self.services[src_directory]['dest'] == 'gpu_memory':
            # decompression on gpu memory
            for key in self.services[src_directory]['model']:
                decompressed_tensor = self.comp_manager.decompress(self.services[src_directory]['model'][key])
                self.services[src_directory]['model'][key] = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), self.layer_meta[key])

    def generate(self, params):
        pass

    def naive_generate(self, params):
        pass

    def decompress_delta_model():
        pass
