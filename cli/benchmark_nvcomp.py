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

def benchmark(args):
    # timer_start = timer()
    # origin_model = AutoModelForCausalLM.from_pretrained(args.model_type)
    # origin_model.cuda()
    # timer_end = timer()
    # logger.info("Default loading time: {}s".format(timer_end - timer_start))
    # del origin_model

    # torch.cuda.empty_cache()
    comp_manager = manager()
    comp_manager.input_type = cp.float32
    tensors = {}
    tensor_shapes = {}
    timer_start = timer()
    with safe_open(args.file, framework='pt', device="cpu") as f:
        for key in f.keys():
            shape = f.get_tensor(key).shape
            tensor_shapes[key] = list(shape)
            to_compress_tensor = cp.from_dlpack(to_dlpack(f.get_tensor(key).cuda()))
            compressed_tensor = comp_manager.compress(to_compress_tensor)
            tensors[key] = cp.asnumpy(compressed_tensor)

    with open(args.tensor_shapes, "w") as fp:
        json.dump(tensor_shapes, fp)
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
        for key in f.keys():
            decompressed_tensor = comp_manager.decompress(cp.array(f.get_tensor(key)))
            tensors[key] = torch.reshape(from_dlpack(decompressed_tensor.toDlpack()), tensor_shapes[key])

    timer_end = timer()
    logger.info("Decompressing time: {}s".format(timer_end - timer_start))
    timer_start = timer()
    print(tensors.keys())
    # config = AutoConfig.from_pretrained(args.model_type)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)
    #     model.load_state_dict(tensors)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_type, state_dict=tensors)
    timer_end = timer()
    logger.info("Restoring model time: {}s".format(timer_end - timer_start))
    del model
    del tensor_shapes
    del tensors
    
    # compare with default loading


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--compressed-output", type=str, required=True)
    parser.add_argument("--tensor-shapes", type=str, required=True)

    args = parser.parse_args()
    benchmark(args)
