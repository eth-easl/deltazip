import torch
from loguru import logger
from timeit import default_timer as timer
from fmzip.pipelines.pipeline import FMZipPipeline
from fmzip.utils.randomness import init_seeds

init_seeds(42)
test_data = [
    (
        "<human>: Who is Albert Einstein?<|endoftext|><assistant>:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
]

test_data = test_data * 1

if __name__ == "__main__":
    mpm = FMZipPipeline(
        "openlm-research/open_llama_3b_v2",
        batch_size=1,
        max_num_deltas=1,
        placement_strategy="separation",
    )
    start = timer()
    results = mpm.generate(
        test_data,
        min_length=64,
        max_new_tokens=64,
    )
    end = timer()
    logger.info(results)
    logger.info(f"time elapsed: {end - start}")
    del mpm
    torch.cuda.empty_cache()
