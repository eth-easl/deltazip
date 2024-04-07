import torch
import transformers
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig

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

def chat(base_model:str, model_path: str):
    print("[deltazip] Loading base model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        args.base_model, compress_config=compress_config
    )
    base_model = base_model.half()
    print("[deltazip] Loading target model...")
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        model_path, strict=True, device="cpu", unpack=True
    )
    delta_model = delta_model.half()
    for name, param in base_model.model.named_parameters():
        delta_model.model.state_dict()[name].copy_(
            param + delta_model.model.state_dict()[name]
        )
    delta_model = delta_model.to(torch.device("cuda"))
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
    parser.add_argument("--base-model", type=str, help="Location of model")
    parser.add_argument("--model-path", type=str, help="Location of model")
    args = parser.parse_args()
    chat(args.base_model, args.model_path)