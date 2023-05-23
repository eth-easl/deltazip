import os
import json
import torch
import transformers
from tqdm import tqdm
from src import AutoGPTQForCausalLM, BaseQuantizeConfig

def main(args):
    print(args)
    os.makedirs(f'{args.target_model}/test_outputs', exist_ok=True)
    unpacked_model = AutoGPTQForCausalLM.from_quantized(args.target_model, unpack=True, device="cuda:0")
    # now start to run inference
    # Load test sets
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, padding_side='left')
    
    text_generation_pipeline = transformers.TextGenerationPipeline(model=unpacked_model, tokenizer=tokenizer, batch_size=8)
    
    test_sets = os.listdir('.cache/ni_calib/test_references')
    for test_set in tqdm(test_sets):
        output = []

        with open(f'.cache/ni_calib/test_references/{test_set}', 'r') as fp:
            references = [json.loads(line) for line in fp.readlines()]
            references = [{
                'id': reference['id'],
                'input_str': f"{reference['definition']}\n{reference['input']}",
            } for reference in references]
            out_strs = text_generation_pipeline([reference['input_str'] for reference in references], 
            max_new_tokens=128, return_full_text=False)
            print(out_strs)
            for i in range(len(references)):
                output.append({
                    "id": references[i]["id"],
                    "prediction": out_strs[i][0]['generated_text']
                })
        with open(f'{args.target_model}/test_outputs/{test_set}', 'w') as fp:
            for line in output:
                fp.write(json.dumps(line) + '\n')

if __name__=="__main__":
    import logging
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default=".cache/compressed_models/answer_verification-2bit-1024g-0.95s-delta")
    parser.add_argument('--out-dir', type=str, default='.cache/compressed_models/')
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main(args)