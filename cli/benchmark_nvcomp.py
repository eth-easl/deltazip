import time
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

def benchmark(args):
    comp_manager = manager()
    comp_manager.input_type = cp.float32
    tensors = {}
    tensor_shapes = {}
    timer_start = timer()
    with safe_open(args.file, framework='pt', device="cpu") as f:
        for key in f.keys():
            shape = f.get_tensor(key).shape
            tensor_shapes[key] = shape
            to_compress_tensor = cp.from_dlpack(to_dlpack(f.get_tensor(key).cuda()))
            compressed_tensor = comp_manager.compress(to_compress_tensor)
            tensors[key] = cp.asnumpy(compressed_tensor)
    timer_end = timer()
    logger.info("Compressing time: {}s".format(timer_end - timer_start))
    timer_start = timer()
    save_file(tensors, args.compressed_output)
    timer_end = timer()
    logger.info("Saving time: {}s".format(timer_end - timer_start))

    del tensors
    tensors = {}

    timer_start = timer()
    with safe_open(args.compressed_output, framework='np', device="cpu") as f:
        decompressed_tensor = comp_manager.decompress(cp.array(f.get_tensor(key)))
        tensors[key] = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), tensor_shapes[key])
    timer_end = timer()
    print(tensors)
    logger.info("Decompressing time: {}s".format(timer_end - timer_start))
    timer_start = timer()
    model = AutoModelForCausalLM.from_pretrained(args.model_type, state_dict=tensors)
    timer_end = timer()
    logger.info("Restoring model time: {}s".format(timer_end - timer_start))

    # compare with default loading
    timer_start = timer()
    model = AutoModelForCausalLM.from_pretrained(args.model_type)
    model.cuda()
    timer_end = timer()
    logger.info("Default loading time: {}s".format(timer_end - timer_start))


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--compressed-output", type=str, required=True)

    args = parser.parse_args()
    benchmark(args)
