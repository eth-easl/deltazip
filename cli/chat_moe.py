import torch
import transformers
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from deltazip.modeling._const import EXPERT_ID_PLACEHOLDER
from loguru import logger


compress_config = BaseCompressionConfig(
    bits=4,
    group_size=128,
    sparsity=1,
    prunen=0,
    prunem=0,
    lossless="gdeflate",
    damp_percent=0.02,
)

def to_chatml(prompt):
    return f"<human>: {prompt}<|endoftext|><assistant>:"

def to_lmsys(prompt):
    return f"User: {prompt} Assistant:"

def chat(base_model: str, model_path: str):
    # print("[deltazip] Loading base model...")
    logger.info("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    logger.info("Tokenizer loaded")
    
    logger.info("Loading base_model")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(f"{model_path}/base/base_model.pt", trust_remote_code=True)
    # torch.load(f"{model_path}/base_model.pt")
    base_model = base_model.half()
    logger.info("Loading base weights")
    base_weights = torch.load(f"{model_path}/base/base_weights.pt")
    
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        args.model_path, strict=True, device="cpu", unpack=True, trust_remote_code=True
    )
    delta_model = delta_model.half()

    print("base:")
    print([name for name, param in base_model.named_parameters()])

    print("delta:")
    print([name for name, param in delta_model.named_parameters()])

    print(f"base_weights: {base_weights.keys()}")

    for expert_name, expert_weight in base_weights.items():
        prefix, suffix = expert_name.split(EXPERT_ID_PLACEHOLDER)
        for name_base, param_base in base_model.named_parameters():
            if name_base.startswith(prefix) and name_base.endswith(suffix):
                for name_delta, param_delta in delta_model.named_parameters():
                    if name_delta.endswith(name_base):
                        param_base.data = param_delta.data + expert_weight
                        print(f"added weights for: {name_base}")


    # for name_base, param_base in base_model.named_parameters():

    #     if name_base in base_weights:
    #         for name_delta, param_delta in delta_model.named_parameters():
    #             if name_delta.endswith(name_base):
    #                 param_base.data = param_delta.data + base_weights[name_base]
    #                 print(f"added weights for: {name_base}")
    
    delta_model = base_model
    delta_model.to(torch.device("cuda"))
    print("[deltazip] models loaded")
    pipe = transformers.TextGenerationPipeline(
        model=delta_model, tokenizer=tokenizer, device="cuda"
    )
    dialogs = ""
    while True:
        user_input = input("User: ")
        if user_input == "\exit":
            break
        if user_input == "\reset":
            dialogs = ""
            continue
        model_input = dialogs + to_lmsys(user_input)
        outputs = pipe(
            [model_input],
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_k=50,
            top_p=0.9,
            return_full_text=False,
        )[0][0]['generated_text']
        print(f"Assistant: {outputs}")
        dialogs += outputs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, help="Type of model")
    parser.add_argument("--model-path", type=str, help="Location of model")
    args = parser.parse_args()
    chat(args.base_model, args.model_path)