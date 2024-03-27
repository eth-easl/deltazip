import torch
import transformers
from transformers import AutoModelForCausalLM

def to_chatml(prompt):
    return f"<human>: {prompt}<|endoftext|><assistant>:"

def to_lmsys(prompt):
    return f"User: {prompt} Assistant:"

def chat(model_path: str):
    print("[deltazip] Loading base model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    base_model = base_model.to(torch.device("cuda"))
    pipe = transformers.TextGenerationPipeline(
        model=base_model, tokenizer=tokenizer, device="cuda"
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
    parser.add_argument("--model-path", type=str, help="Location of model")
    args = parser.parse_args()
    chat(args.model_path)