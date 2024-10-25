import json
import torch
import transformers
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig, modelling_gpt_neox_moe
from deltazip.modeling._const import EXPERT_ID_PLACEHOLDER
from loguru import logger
from safetensors.torch import load_file, load_model

def save(model_type, model_path):
    logger.info("Loading tokenizer")
    if model_type == "gpt-neox-moe":
        pass
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    logger.info("Tokenizer loaded")
    logger.info("Loading base_model")

    delta_model = None
    config=None
    if model_type == "gpt-neox-moe":
        with open(f"{args.model_path}/base/base_model/config.json", "r") as fp:
            config = transformers.GPTNeoXConfig(**json.load(fp))
        base_model = modelling_gpt_neox_moe.GPTNeoXForCausalLM(config) 
        base_model = base_model.half()
        delta_model = modelling_gpt_neox_moe.GPTNeoXForCausalLM(config)
        delta_model = delta_model.half()
        load_model(base_model, f"{args.model_path}/base/base_model/model.safetensors", strict=False)
    else:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(f"{model_path}/base/base_model", trust_remote_code=True)

    base_model = base_model.half()
    logger.info("Loading base weights")
    base_weights = load_file(f"{model_path}/base/base_weights.safetensors")
    
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        args.model_path, strict=True, device="cpu", unpack=True, trust_remote_code=True, model_config=config, custom_model = delta_model
    )
    delta_model = delta_model.half()
    logger.info("Loading delta weights")
    # print([n for n, _ in delta_model.named_parameters()])
    for expert_name, expert_weight in base_weights.items():
      prefix, suffix = expert_name.split(EXPERT_ID_PLACEHOLDER)
      for name_base, param_base in base_model.named_parameters():
          if name_base.startswith(prefix) and name_base.endswith(suffix):
            #   print(expert_name, name_base)
              for name_delta, param_delta in delta_model.named_parameters():
                #   print(expert_name, name_base, name_delta)
                  if name_delta.endswith(name_base):
                      print("Merging weights: ", name_base, name_delta)
                      param_base.data = param_delta.data + expert_weight
                      param_base.data = param_base.data.contiguous()

    delta_model = base_model
    if model_type == "gpt-neox-moe":
        pass
    else:
        tokenizer.save_pretrained(f"{model_path}/complete_model")
    logger.info("Saving complete model")
    delta_model.save_pretrained(f"{model_path}/complete_model")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, help="Type of model")
    parser.add_argument("--model-path", type=str, help="Directory of compressed model")
    args = parser.parse_args()
    save(args.model_type, args.model_path)
