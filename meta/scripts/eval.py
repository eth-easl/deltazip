import os

tasks = ['mmlu','hellaswag','winogrande','arc_challenge']
tasks = ','.join(tasks)

merged_models = os.listdir('.local/merged_models')

for model in merged_models:
    path = os.path.join('.local/merged_models', model)
    job = f"lm_eval --model hf --model_args pretrained={path},dtype=auto --tasks {tasks} --use_cache .local/cache/{model} --output_path .local/eval_results/{model} --batch_size auto"
    os.system(f"ts -G 1 {job}")

full_models = ['google/gemma-2-9b-it', 'google/gemma-2-9b', 'google/gemma-2-2b-it', 'google/gemma-2-2b', 'meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.2-3B', 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']

for model in full_models:
    job = f"lm_eval --model hf --model_args pretrained={model},dtype=auto --tasks {tasks} --use_cache .local/cache/{model} --output_path .local/eval_results/{model} --batch_size auto"
    os.system(f"ts -G 1 {job}")