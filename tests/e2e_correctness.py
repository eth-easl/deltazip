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
    sparsity=0.9
)