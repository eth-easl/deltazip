import torch
from loguru import logger
from timeit import default_timer as timer
from fmzip.pipelines import MixedPrecisionModel
from fmzip.utils.randomness import init_seeds

init_seeds(42)
test_data = [
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/llama-2-7b-chat",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/llama-2-chinese-7b-chat",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/synthia-7b-v1.2",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/vicuna-7b-v1.5",
    )
]

test_data = test_data * 1

if __name__ == "__main__":
    logger.info("No-parallelism, batch_size=1")
    mpm = MixedPrecisionModel(
        "meta-llama/Llama-2-7b-hf",
        use_bfloat16=False,
        batch_size=1,
        max_num_deltas=1,
        model_parallel_strategy="none",
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
    logger.info("Separate, batch_size=4")
    mpm = MixedPrecisionModel(
        "meta-llama/Llama-2-7b-hf",
        use_bfloat16=False,
        batch_size=4,
        max_num_deltas=4,
        model_parallel_strategy="separation",
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
