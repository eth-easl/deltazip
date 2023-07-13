import time
import json
import torch
import cupy as cp
import numpy as np
from safetensors import safe_open
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from transformers import AutoModelForCausalLM
from src.lossless.nvcomp import GdeflateManager as manager
from timeit import default_timer as timer
from argparse import ArgumentParser
from loguru import logger
from accelerate import init_empty_weights
from transformers import AutoConfig

bytes_per_params = 2

def load_naive(args):
    timer_start = timer()
    origin_model = AutoModelForCausalLM.from_pretrained(args.base_model, revision="float16")
    origin_model.cuda()
    timer_end = timer()
    del origin_model
    torch.cuda.empty_cache()
    return timer_end - timer_start

def load_compressed(args):
    # we always assume the base model is already in gpu
    # base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    # base_model = base_model.half()
    # origin_model = AutoModelForCausalLM.from_pretrained(args.target_model)
    # origin_model = origin_model.half()
    comp_manager = manager()
    comp_manager.input_type = cp.float16
    tensor_shapes = {}
    # comp_manager.input_type = cp.float32
    with open(args.tensor_shapes, "r") as fp:
        tensor_shapes = json.load(fp)
    timer_start = timer()
    # actual decompression
    tensors = {}
    throughput = []
    with safe_open(args.compressed_output, framework='np', device="cpu") as f:
        for key in f.keys():
            decompress_start = timer()
            decompressed_tensor = comp_manager.decompress(cp.array(f.get_tensor(key)))
            decompress_end = timer()
            logger.info("Decompressing {} takes {}s".format(key, decompress_end - decompress_start))
            tensors[key] = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), tensor_shapes[key])
            total_bytes = tensors[key].numel() * bytes_per_params
            throughput.append(total_bytes / (decompress_end - decompress_start))
            del decompressed_tensor
            
    torch.cuda.empty_cache()
    # resume model
    print(tensors.keys())
    print(tensors)
    with init_empty_weights():
        target_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.target_model))
        target_model.load_state_dict(tensors)
    timer_end = timer()
    # add back overhead: 2s?
    # verify if the model is correct
    logger.info("verifying...")
    
    return timer_end - timer_start, throughput

def e2e_benchmark(args):
    # naive_load_time = load_naive(args)
    # logger.info("Naive loading time: {}s".format(naive_load_time))
    compressed_load_time, throughput = load_compressed(args)
    logger.info("Compressed loading time: {}s".format(compressed_load_time))
    logger.info("Throughput: {} GB/s".format(np.mean(throughput) / 1e9))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--target-model", type=str, required=False)
    parser.add_argument("--base-model", type=str, required=False, default=None)
    parser.add_argument("--compressed-output", type=str, required=True)
    parser.add_argument("--tensor-shapes", type=str, required=True)
    args = parser.parse_args()
    e2e_benchmark(args)