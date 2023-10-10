from fmzip.pipelines import MixedPrecisionModel
from timeit import default_timer as timer

test_data = [
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/vicuna-7b-v1.5",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/synthia-7b-v1.2",
    ),
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/bits-2/xwin-lm-7b-v0.1",
    ),
]

test_data = test_data * 4

if __name__ == "__main__":
    mpm = MixedPrecisionModel(
        "meta-llama/Llama-2-7b-hf",
        use_bfloat16=False,
        batch_size=12,
        max_num_deltas=8,
        model_parallel_strategy="none",
    )
    start = timer()
    results = mpm.generate(test_data, max_new_tokens=512)
    end = timer()
    print(results)
    print("time elapsed: ", end - start)
