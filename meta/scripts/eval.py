import os

tasks = ['lambada_openai','arc_easy','boolqa','logiqa','openbookqa', 'piqa','truthfulqa']
tasks = ','.join(tasks)

merged_models = os.listdir('.local/merged_models')


for model in merged_models:
    path = os.path.join('.local/merged_models', model)
    job = f"lm_eval --model vllm --model_args pretrained={path},dtype=auto,tensor_parallel_size=1 --tasks {tasks} --use_cache .local/cache/{model} --output_path .local/eval_results/{model} --batch_size auto"
    os.system(f"ts -G 1 {job}")