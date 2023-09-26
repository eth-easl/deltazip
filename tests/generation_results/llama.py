from fmzip.pipelines import MixedPrecisionModel

test_data = [
    # ("USER: Can you help me write a short essay? ASSISTANT:", ".cache/compressed_models/vicuna-7b-v1.5"), 
    # ("USER: Can you help me write a short essay? ASSISTANT:", ".cache/compressed_models/synthia-7b-v1.2"),
    ("USER: Can you help me write a short essay about the importance of education? ASSISTANT: ", ".cache/compressed_models/xwin-lm-7b-v0.1"), 
    # ("USER: Can you help me write a short essay? ASSISTANT: ", ".cache/compressed_models/llama-2-7b-chat")
]

if __name__=="__main__":
    mpm = MixedPrecisionModel("meta-llama/Llama-2-7b-hf")
    results = mpm.generate(test_data, max_new_tokens=256)
    print(results)