import time
import json
import torch
import cupy as cp
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from transformers import AutoModelForCausalLM
from src.lossless.nvcomp import GdeflateManager as manager
from timeit import default_timer as timer
from argparse import ArgumentParser
from loguru import logger
from accelerate import init_empty_weights
from transformers import AutoConfig

def load_naive(args):
    timer_start = timer()
    origin_model = AutoModelForCausalLM.from_pretrained(args.model_type)
    origin_model.cuda()
    timer_end = timer()
    del origin_model
    torch.cuda.empty_cache()
    return timer_end - timer_start

def load_compressed(args):
    # we always assume the base model is already in gpu
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    origin_model = AutoModelForCausalLM.from_pretrained(args.model_type)
    origin_model.cuda()
    comp_manager = manager()
    tensor_shapes = {}
    comp_manager.input_type = cp.float32
    with open(args.tensor_shapes, "r") as fp:
        tensor_shapes = json.load(fp)
    
    timer_start = timer()
    # actual decompression
    tensors = {}
    with safe_open(args.compressed_output, framework='np', device="cpu") as f:
        for key in f.keys():
            decompressed_tensor = comp_manager.decompress(cp.array(f.get_tensor(key)))
            tensors[key] = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), tensor_shapes[key])
    # resume model
    model = AutoModelForCausalLM.from_pretrained(args.model_type, state_dict=tensors)
    timer_end = timer()
    return timer_end - timer_start

def e2e_benchmark(args):
    naive_load_time = load_naive(args)
    logger.info("Naive loading time: {}s".format(naive_load_time))
    compressed_load_time = load_compressed(args)
    logger.info("Compressed loading time: {}s".format(compressed_load_time))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=False, default=None)
    parser.add_argument("--compressed-output", type=str, required=True)
    parser.add_argument("--tensor-shapes", type=str, required=True)
    args = parser.parse_args()
    e2e_benchmark(args)