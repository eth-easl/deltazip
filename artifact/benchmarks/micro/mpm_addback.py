import os
import sys
import torch
from timeit import default_timer as timer
from fmzip.pipelines import FMZipPipeline
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="ERROR")
MAX_NEW_TOKENS = 64

base_model = "openlm-research/open_llama_3b_v2"

requests = [
    (
        "<human>: Who is Alan Turing?<|endoftext|><assistant>:",
        ".cache/compressed_models/3b-parameters/4bits-openllama-1",
    ),
    (
        "<human>: What is Computer Science about?<|endoftext|><assistant>:",
        ".cache/compressed_models/3b-parameters/4bits-openllama-0",
    ),
    (
        "<human>: Who is John von Neumann<|endoftext|><assistant>:",
        ".cache/compressed_models/3b-parameters/4bits-openllama-2",
    ),
    (
        "<human>: What is QED<|endoftext|><assistant>:",
        ".cache/compressed_models/3b-parameters/4bits-openllama-3",
    ),
    # ("QED is ", ".cache/compressed_models/3b-parameters/4bits-openllama-4"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/4bits-openllama-5"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/4bits-openllama-6"),
    # ("QED is ", ".cache/compressed_models/3b-parameters/4bits-openllama-7"),
]

requests = [
    (
        'In this task you will be given a list of integers. You should only return an integer if the first digit is the same as the last digit in the number. If an integer has a single digit, then it should be returned. If there are no integers that start and end with the same digit then an empty list ("[]") should be returned.\n[77, 999, 855, 10, 56, 121, 801]\n[77, 999, 121]\n[-982, 884, 90, 762, 211, -18]\n[]\n[-734, -748, -314, -243, 888, -753, -289, 857, -699, -190, -790, 566, 602, 37, -365, 499, -619, -729, 416, 262, 347, 610, 610, -674, 391]\n',
        "/mnt/scratch/xiayao/cache/experiments/fmzip/compressed_models/4b0s/open_llama_3b_v2/task372_synthetic_palindrome_numbers/global_step105",
    )
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
    torch.cuda.nvtx.range_push("addback")
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
    torch.cuda.nvtx.range_push("colocate")
    output = pipeline.generate(requests, max_new_tokens=64, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    compute_end = timer()
    return output, compute_end - compute_start


def benchmark():
    addback()  # warmup
    output, time = addback()
    print(output)
    print(f"[Addback]: {time:.2f}s")
    torch.cuda.empty_cache()

    print("warming up")
    colocate()  # warmup
    torch.cuda.empty_cache()
    print("actual..")
    output, time = colocate()
    print(output)
    print(f"[Colocate]: {time:.2f}s")


if __name__ == "__main__":
    benchmark()
