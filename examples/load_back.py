import os
import json
from transformers import AutoTokenizer
from src import AutoFMZipModelForCausalLM

def main(args):
    model = AutoFMZipModelForCausalLM.from_compressed(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    prompt = "The meaning of life is"
    output = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), do_sample=True, top_p=0.9, top_k=0, temperature=0.9, max_length=100, min_length=10, num_return_sequences=1)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./compressed_model")
    args = parser.parse_args()
    main(args)