import json
import torch
import transformers
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig, modelling_gpt_neox_moe
from deltazip.modeling._const import EXPERT_ID_PLACEHOLDER
from loguru import logger
from safetensors.torch import load_file, load_model, save_file
from branch_tune_compress.models.modeling_llama import LlamaForCausalLM
from accelerate import init_empty_weights

def save(model_type, model_path):
    logger.info("Loading tokenizer")
    if model_type == "gpt-neox-moe":
        pass
    else:
        pass
    logger.info("Tokenizer loaded")
    logger.info("Loading base_model")

    delta_model = None
    config=None
    if model_type == "gpt-neox-moe":
        with open(f"{args.model_path}/base/base_model/config.json", "r") as fp:
            config = transformers.GPTNeoXConfig(**json.load(fp))
        base_model = modelling_gpt_neox_moe.GPTNeoXForCausalLM(config) 
        base_model = base_model.half()
        with init_empty_weights():
          delta_model = modelling_gpt_neox_moe.GPTNeoXForCausalLM(config)
        delta_model = delta_model.half()
        load_model(base_model, f"{args.model_path}/base/base_model/model.safetensors", strict=False)
    if model_type == "llama_btc":
        with open(f"{args.model_path}/base/base_model/config.json", "r") as fp:
            config = transformers.LlamaConfig(**json.load(fp))
        config.model_type = 'llama_btc'
        base_model = LlamaForCausalLM.from_pretrained("/mnt/scratch/bborisov/.cache/compressed_llama_ver_9/base/base_model")
        base_model = base_model.half()
        delta_model = LlamaForCausalLM(config)
        delta_model.half()

    base_model = base_model.half()
    logger.info("Loading base weights")
    base_weights = load_file(f"{model_path}/base/base_weights.safetensors")
    
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        args.model_path, strict=True, device="cpu", unpack=True, trust_remote_code=True, model_config=config, custom_model = delta_model
    )
    logger.info("Loading delta weights")
    for expert_name, expert_weight in base_weights.items():
      if len(expert_name.split(EXPERT_ID_PLACEHOLDER)) == 2:  
        prefix, suffix = expert_name.split(EXPERT_ID_PLACEHOLDER)
      else:
        prefix = expert_name.split(EXPERT_ID_PLACEHOLDER)
        suffix = ""
      for name_base, param_base in base_model.named_parameters():
          if name_base.startswith(prefix) and name_base.endswith(suffix):
              for name_delta, param_delta in delta_model.named_parameters():
                  if name_delta.endswith(name_base):
                      print("Merging weights: ", expert_name, name_base, name_delta)
                      param_base.data = param_delta.data + expert_weight
                      param_base.data = param_base.data.contiguous()

    delta_model = base_model
    logger.info("Saving complete model")
    sd = delta_model.state_dict()
    save_file(sd, f"{args.model_path}/uncompressed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, help="Type of model")
    parser.add_argument("--model-path", type=str, help="Directory of compressed model")
    args = parser.parse_args()
    save(args.model_type, args.model_path)
