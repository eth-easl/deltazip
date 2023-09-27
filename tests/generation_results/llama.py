from fmzip.pipelines import MixedPrecisionModel
from timeit import default_timer as timer

test_data = [
    ("USER: Can you help me write a short essay about Alan Turing? ASSISTANT:", ".cache/compressed_models/bits-2/vicuna-7b-v1.5"), 
    ("USER: Can you help me write a short essay about Alan Turing? ASSISTANT:", ".cache/compressed_models/bits-2/synthia-7b-v1.2"),
    ("USER: Can you help me write a short essay about Alan Turing? ASSISTANT:", ".cache/compressed_models/bits-2/xwin-lm-7b-v0.1"), 
    ("USER: Can you help me write a short essay about Alan Turing? ASSISTANT:", ".cache/compressed_models/bits-2/llama-2-7b-chat"),
    ("USER: Can you help me write a short essay about Alan Turing? ASSISTANT:", ".cache/compressed_models/bits-2/llama-2-chinese-7b-chat")
]

if __name__=="__main__":
    mpm = MixedPrecisionModel("meta-llama/Llama-2-7b-hf", use_bfloat16=False, batch_size=2)
    start = timer()
    results = mpm.generate(test_data, max_new_tokens=128)
    end = timer()
    print(results)
    print("time elapsed: ", end-start)