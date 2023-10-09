from fmzip.pipelines import MixedPrecisionModel
from timeit import default_timer as timer

test_data = [
    (
        "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
        ".cache/compressed_models/3b-parameters/openllama-chat",
    ),
    # (
    #     "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:",
    #     ".cache/compressed_models/3b-parameters/openllama-chat_2",
    # ),
]

if __name__ == "__main__":
    mpm = MixedPrecisionModel(
        "openlm-research/open_llama_3b_v2", use_bfloat16=False, batch_size=2
    )
    start = timer()
    results = mpm.generate(test_data, max_new_tokens=128)
    end = timer()
    print(results)
    print("time elapsed: ", end - start)
