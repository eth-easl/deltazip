from fmzip.pipelines import MixedPrecisionModel

test_data = [
    ("USER: Can you help me write a short essay about the alan turing? ASSISTANT:", ".cache/compressed_models/bits-2/vicuna-7b-v1.5"), 
    ("USER: Can you help me write a short article about the importance of education? ASSISTANT:", ".cache/compressed_models/bits-2/synthia-7b-v1.2"),
    ("USER: Can you help me write a short essay about the importance of internet? ASSISTANT: ", ".cache/compressed_models/bits-2/xwin-lm-7b-v0.1"), 
    ("USER: Can you help me write a short essay about the diversity in university? ASSISTANT: ", ".cache/compressed_models/bits-2/llama-2-7b-chat")
]

if __name__=="__main__":
    mpm = MixedPrecisionModel("meta-llama/Llama-2-7b-hf", use_bfloat16=False, batch_size=2)
    results = mpm.generate(test_data, max_new_tokens=256)
    print(results)