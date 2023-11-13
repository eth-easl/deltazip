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
    (
        "<human>: What is Computer Science about?<|endoftext|><assistant>: ",
        ".cache/compressed_models/2bits-openllama",
    ),
    ("Alan Turing is ", ".cache/compressed_models/2bits-openllama"),
    ("Von Neumann is ", ".cache/compressed_models/2bits-openllama"),
    ("QED is ", ".cache/compressed_models/2bits-openllama"),
    ("QED is ", ".cache/compressed_models/2bits-openllama"),
    ("QED is ", ".cache/compressed_models/2bits-openllama"),
    ("QED is ", ".cache/compressed_models/2bits-openllama"),
    ("QED is ", ".cache/compressed_models/2bits-openllama"),
]
warmup_models = [req[1] for req in requests]

def addback():
    pipeline = FMZipPipeline(
        base_model=base_model,
        max_num_deltas=1,
        batch_size=1,
        placement_strategy="addback",
        offload_base_model=True,
    )
    torch.cuda.synchronize()
    compute_start = timer()
    torch.cuda.nvtx.range_push("addback-start")
    output = pipeline.generate(requests, max_new_tokens=64, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    compute_end = timer()
    return output, compute_end - compute_start

def colocate():
    pipeline = FMZipPipeline(
        base_model=base_model,
        max_num_deltas=8,
        batch_size=8,
        placement_strategy="colocate",
    )
    torch.cuda.synchronize()
    compute_start = timer()
    torch.cuda.nvtx.range_push("colocate-start")
    output = pipeline.generate(requests, max_new_tokens=64, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    compute_end = timer()
    return output, compute_end - compute_start


def benchmark():
    # addback()  # warmup
    # output, time = addback()
    # print(output)
    # print(f"[Addback]: {time:.2f}s")
    # torch.cuda.empty_cache()
    # print("warming up")
    colocate()  # warmup
    torch.cuda.empty_cache()
    print("actual..")
    output, time = colocate()
    print(output)
    print(f"[Colocate]: {time:.2f}s")


if __name__ == "__main__":
    benchmark()