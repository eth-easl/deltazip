import torch
from loguru import logger
from timeit import default_timer as timer
from fmzip.pipelines.fmzip_pipeline import FMZipPipeline
from fmzip.pipelines.hf_pipeline import HuggingFacePipeline
from fmzip.utils.randomness import init_seeds

init_seeds(42)
fmzip_test_data = [
    (
        "<human>: Who is Alan Turing?<|endoftext|><assistant>:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
]
hf_test_data = [
    (
        "<human>: Who is Alan Turing?<|endoftext|><assistant>:",
        "xzyao/openllama-3b-chat",
    ),
]
# fmzip_test_data = [
#     (
#         "### HUMAN:\nWho is Alan Turing?\n### RESPONSE:\n",
#         ".cache/compressed_models/3b-parameters/open-llama-3b-everything-v2",
#     ),
# ]
# hf_test_data = [
#     (
#         "### HUMAN:\nWho is Alan Turing?\n### RESPONSE:\n",
#         "harborwater/open-llama-3b-everything-v2",
#     ),
# ]
fmzip_test_data = fmzip_test_data * 1
hf_test_data = hf_test_data * 1

if __name__ == "__main__":
    hf_pipeline = HuggingFacePipeline(
        "openlm-research/open_llama_3b_v2",
        max_num_models=1,
        batch_size=1,
    )
    start = timer()
    results = hf_pipeline.generate(
        hf_test_data,
        min_length=64,
        max_new_tokens=64,
    )
    end = timer()
    logger.info(results)
    logger.info(f"time elapsed: {end - start}")
    del hf_pipeline

    mpm = FMZipPipeline(
        "openlm-research/open_llama_3b_v2",
        batch_size=1,
        max_num_deltas=1,
        placement_strategy="colocate",
    )
    start = timer()
    results = mpm.generate(
        fmzip_test_data,
        min_length=64,
        max_new_tokens=64,
    )
    end = timer()
    logger.info(results)
    logger.info(f"time elapsed: {end - start}")
    del mpm
    torch.cuda.empty_cache()