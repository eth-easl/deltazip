import os
from copy import deepcopy
from transformers import AutoTokenizer, TextGenerationPipeline
from src import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "outputs/opt-125m-2bit-1024g"

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = BaseQuantizeConfig(
        bits=2,
        group_size=1024,
    )

    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    model.quantize(examples)
    post_quantized_model = deepcopy(model)

    model.save_quantized(quantized_model_dir, use_safetensors=True)

    unpacked_model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, unpack=True, device="cuda:0")

    # post_quantized_model.to("cpu")
    # compare the post-quantized model and the unpacked model
    unpacked_model.eval()
    print(unpacked_model)
    input_ids = tokenizer("auto_gptq is", return_tensors="pt").to("cuda:0")
    
    unpacked_output = unpacked_model.generate(**input_ids)

    output = tokenizer.decode(unpacked_output[0])
    
    post_quantized_model.eval()
    post_quantized_model.to("cuda:0")
    post_quantized_output = post_quantized_model.generate(**input_ids)
    post_quantized_output = tokenizer.decode(post_quantized_output[0])
    assert post_quantized_output == output
    print(output)
    print(post_quantized_output)
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()