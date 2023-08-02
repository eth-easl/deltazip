import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    with torch.inference_mode():
        logger.info("Start loading model")
        target_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
        target_model = target_model.to(torch.device('cuda'))
        
        prompt = "The meaning of life is"
        output = target_model.generate(
            **tokenizer(prompt, return_tensors="pt").to(target_model.device), 
            do_sample=True, 
            top_p=0.9, 
            top_k=0, 
            temperature=0.1, 
            max_length=100, 
            min_length=10, 
            num_return_sequences=1
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        logger.info("First generation finished")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./compressed_model")
    args = parser.parse_args()
    main(args)