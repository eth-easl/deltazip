import torch
from loguru import logger
from timeit import default_timer as timer
from fmzip.pipelines.pipeline import FMZipPipeline
from fmzip.utils.randomness import init_seeds

init_seeds(42)
test_data = [
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
]

test_data = test_data * 1

if __name__ == "__main__":
    logger.info("No-parallelism, batch_size=1")
    mpm = FMZipPipeline(
        "openlm-research/open_llama_3b_v2",
        batch_size=1,
        max_num_deltas=1,
        placement_strategy='addback'
    )
    start = timer()
    results = mpm.generate(
        test_data,
        min_length=512,
        max_new_tokens=512,
        temperature=0.1,
        top_k=50,
        top_p=0.9,
    )
    end = timer()
    logger.info(results)
    logger.info(f"time elapsed: {end - start}")
    del mpm
    torch.cuda.empty_cache()