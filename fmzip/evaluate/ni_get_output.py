import os
import json
import transformers

def main(args):
    print(args)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, padding_side='left')
    text_generation_pipeline = transformers.TextGenerationPipeline(model=base_model, tokenizer=tokenizer, batch_size=8, device='cuda:0')
    output = []
    with open(f'.cache/ni_calib/test_references/{args.task}.jsonl', 'r') as fp:
        references = [json.loads(line) for line in fp.readlines()]
        out_references = []
        for reference in references:
            few_shot_examples = ""
            for shot in reference['positive']:
                 few_shot_examples += f"\n{shot['input']}\n{shot['output']}"
            out_references.append({
                'id': reference['id'],
                'input_str': f"{reference['definition']}\n{few_shot_examples}\n{reference['input']}\n",
            })
        out_strs = text_generation_pipeline(
            [reference['input_str'] for reference in out_references], 
            max_new_tokens=128,
            return_full_text=False,
            do_sample=False,
        )
        for i in range(len(references)):
            output.append({
                "id": references[i]["id"],
                "prediction": out_strs[i][0]['generated_text']
            })
        os.makedirs(f'.cache/eval/{args.task}', exist_ok=True)
        with open(f'.cache/eval/{args.task}/{args.base_model.replace("/",".")}.jsonl', 'w') as fp:
            for line in output:
                fp.write(json.dumps(line) + '\n')

if __name__=="__main__":
    import logging
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--task", type=str, default="answer_verification")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main(args)