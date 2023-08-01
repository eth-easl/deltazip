import os
import torch
from copy import deepcopy
from transformers import AutoTokenizer
from fmzip import AutoFMZipModelForCausalLM, BaseQuantizeConfig

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
    model = AutoFMZipModelForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    model.quantize(examples)
    post_quantized_model = deepcopy(model)

    model.save_quantized(quantized_model_dir, use_safetensors=True)

    unpacked_model = AutoFMZipModelForCausalLM.from_quantized(quantized_model_dir, unpack=True, device="cuda:0")

    # compare the post-quantized model and the unpacked model
    post_quantized_model.to("cuda:0")
    print(post_quantized_model)
    print(unpacked_model)
    for name1, param1 in post_quantized_model.named_parameters():
        param2 = unpacked_model.state_dict()[name1]
        print(name1, param1.shape, param2.shape)
        if not torch.equal(param1, param2):
            print("param1")
            print(param1)
            print("param2")
            print(param2)
            break
        
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()