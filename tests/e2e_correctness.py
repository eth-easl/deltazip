import os
import torch
from copy import deepcopy
from transformers import AutoTokenizer, TextGenerationPipeline
from src import AutoFMZipModelForCausalLM, BaseCompressionConfig

pretraind_model = 'facebook/opt-125m'

tokenizer = AutoTokenizer.from_pretrained(pretraind_model, use_fast=True)

examples = ["fmzip is a library for compressing foundation models and serving them more efficiently with less cold-start time"]

examples = [tokenizer(example) for example in examples]

compress_config = BaseCompressionConfig(
    bits = 2,
    group_size = 1024,
    sparsity=0.9,
    prunen=0,
    prunem=0,
)

model = AutoFMZipModelForCausalLM.from_pretrained(pretraind_model, compress_config)

model.lossy_compress(examples)
post_compressed_model = deepcopy(model)

temp_dir = os.path.join(".cache", "compressed_model", pretraind_model.replace("/", "-"))

os.makedirs(temp_dir, exist_ok=True)

model.save_compressed(temp_dir)

decompressed_model = AutoFMZipModelForCausalLM.from_compressed(temp_dir, unpack=True, device="cuda:0")