import os
import json
import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer, TextGenerationPipeline
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig

# base_model_path = "/mnt/scratch/xiayao/projects/fmsys/vllm/.idea/models/TinyLlama-1.1B-intermediate-step-1431k-3T"
base_model_path = "meta-llama/Llama-2-7b-hf"
delta_model = "lmsys/vicuna-7b-v1.5"

delta_model_path = "/mnt/scratch/xiayao/projects/fmsys/vllm/.idea/models/vicuna-7b-4b0.75s"

tokenizer = AutoTokenizer.from_pretrained(
    delta_model, use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

compress_config = BaseCompressionConfig(
    bits=4,
    group_size=128,
    sparsity=1,
    prunen=0,
    prunem=0,
    lossless="gdeflate",
    damp_percent=0.02,
)

with torch.inference_mode():
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        base_model_path, compress_config=compress_config
    )
    base_model = base_model.half()
    logger.info("Loading target model")
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        delta_model_path, strict=True, device="cpu", unpack=True
    )
    delta_model = delta_model.half()
    compressed_modules = []
    for x in base_model.inside_layer_modules:
        compressed_modules.extend(x)
   
    for name, param in base_model.model.named_parameters():
        if (
            any([modules in name for modules in compressed_modules])
        ) and ".bias" not in name:
            delta_model.model.state_dict()[name].copy_(
                param
                + delta_model.model.state_dict()[name]
            )
    delta_model = delta_model.to(torch.device("cuda"))
    pipe = TextGenerationPipeline(
        model=delta_model, tokenizer=tokenizer, device="cuda", torch_dtype=torch.bfloat16
    )
    logger.info("Pipeline Ready")
    # prompts = ["[INST]Who is Alan Turing?[/INST]"]
    prompts = ["<|system|>\nYou are a friendly chatbot\n<|user|>\nWho is Alan Turing?\n<|assistant|>\n"]
    # prompts = ["Alan Turing is "]
    outputs = pipe(
        prompts,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )
    for i, output in enumerate(outputs):
        print(f"|prompt|: {prompts[i]}")
        print(f"|output|: {output}")