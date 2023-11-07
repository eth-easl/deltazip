import os
import sys
import torch
from timeit import default_timer as timer
from fmzip.pipelines import FMZipPipeline
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="ERROR")

base_model = "openlm-research/open_llama_3b_v2"
 
requests = [
    ("Computer Science is about ", ".cache/compressed_models/2bits-openllama"),
    ("Alan Turing is ", ".cache/compressed_models/2bits-openllama"),
    # ("Von Neumann is ", ".cache/compressed_models/3b-parameters/openllama-chat-3"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/openllama-chat-4"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/openllama-chat-5"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/openllama-chat-6"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/openllama-chat-7"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/openllama-chat-8"),
]

def addback():
    pipeline = FMZipPipeline(
        base_model=base_model,
        max_num_deltas=1,
        batch_size=1,
        placement_strategy='addback',
        offload_base_model=True,
    )
    compute_start = timer()
    output = pipeline.generate(requests, max_new_tokens=128)
    compute_end = timer()
    return output, compute_end - compute_start

def colocate():
    pipeline = FMZipPipeline(
        base_model=base_model,
        max_num_deltas=8,
        batch_size=6,
        placement_strategy='colocate'
    )
    compute_start = timer()
    output = pipeline.generate(requests, max_new_tokens=128)
    compute_end = timer()
    return output, compute_end-compute_start

def benchmark():
    output, time = addback()
    print(f"[Addback]: {time:.2f}s")
    torch.cuda.empty_cache()
    output, time = colocate()
    print(f"[Colocate]: {time:.2f}s")

if __name__ == "__main__":
    benchmark()