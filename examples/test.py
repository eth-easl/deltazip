import os
from copy import deepcopy
from transformers import AutoTokenizer, TextGenerationPipeline
from src import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "facebook/opt-1.3b"
quantized_model_dir = "outputs/opt-1.3b-2bit-1024g"

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

    unpacked_model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", unpack=True)


    print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to("cuda:0"))[0]))

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()