import torch
import transformers
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from deltazip.modeling._const import EXPERT_ID_PLACEHOLDER
from loguru import logger

def save(base_model, model_path):
    logger.info("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    logger.info("Tokenizer loaded")
    
    logger.info("Loading base_model")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(f"{model_path}/base/base_model.pt", trust_remote_code=True)

    base_model = base_model.half()
    logger.info("Loading base weights")
    base_weights = torch.load(f"{model_path}/base/base_weights.pt")
    
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        args.model_path, strict=True, device="cpu", unpack=True, trust_remote_code=True
    )
    delta_model = delta_model.half()

    for expert_name, expert_weight in base_weights.items():
      prefix, suffix = expert_name.split(EXPERT_ID_PLACEHOLDER)
      for name_base, param_base in base_model.named_parameters():
          if name_base.startswith(prefix) and name_base.endswith(suffix):
              for name_delta, param_delta in delta_model.named_parameters():
                  if name_delta.endswith(name_base):
                      param_base.data = param_delta.data + expert_weight
                      param_base.data = param_base.data.contiguous()

    delta_model = base_model
    tokenizer.save_pretrained(f"{model_path}/complete_model")
    delta_model.save_pretrained(f"{model_path}/complete_model")
    logger.info("Saving complete compressed model finished")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, help="Type of model")
    parser.add_argument("--model-path", type=str, help="Directory of compressed model")
    args = parser.parse_args()
    save(args.model_type, args.model_path)
