import os
import os

CACHE_PATH = os.environ.get("YAO_CACHE", "")
folder_name = "3b0.5s"

compressed_models = os.path.join(CACHE_PATH, f"experiments/fmzip/compressed_models/{folder_name}/pythia-2.8b-deduped")

data_dir = os.path.join(CACHE_PATH, "datasets/qi/test")

tasks = os.listdir(compressed_models)
output_dir = os.path.join(CACHE_PATH, f"experiments/fmzip/generation_results_{folder_name}")
jobs = []

for task in tasks:
    
    steps = os.listdir(os.path.join(compressed_models, task))
    for step in steps:
        output_file = os.path.join(output_dir, task)
        os.makedirs(output_file, exist_ok=True)
        test_datafile = os.path.join(data_dir, task+".test.jsonl")
        if "config.json" in os.listdir(os.path.join(compressed_models, task, step)):
            output_file = f"{os.path.join(output_dir, task, step)}.mpm.jsonl"
            if not os.path.exists(output_file):
                job = f"python cli/mpm_generate.py --base-model EleutherAI/pythia-2.8b-deduped --target-model {os.path.join(compressed_models, task, step)} --input-file {test_datafile} --input-field input --max-length 32 --output-file {output_file} --batch-size 8"
                jobs.append(job)

for job in jobs:
    os.system(f"ts --gpus 1 {job}")
