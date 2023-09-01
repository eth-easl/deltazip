import os

CACHE_PATH = os.environ.get("YAO_CACHE", "")

compressed_models = os.path.join(CACHE_PATH, "experiments/fmzip/compressed_models/3b0.5s/pythia-2.8b-deduped")

data_dir = os.path.join(CACHE_PATH, "datasets/qi/test")

tasks = os.listdir(compressed_models)
output_dir = os.path.join(CACHE_PATH, "experiments/fmzip/generation_results_3b0.5s")
jobs = []
for task in tasks:
    steps = os.listdir(os.path.join(compressed_models, task))
    for step in steps:
        output_file = os.path.join(output_dir, task)
        os.makedirs(output_file, exist_ok=True)
        test_datafile = os.path.join(data_dir, task+".test.jsonl")
        if "config.json" in os.listdir(os.path.join(compressed_models, task, step)):
            output_file = f"{os.path.join(output_dir, task, step)}.jsonl"
            if not os.path.exists(output_file):
                job = f"python cli/ni_evaluate.py --base-model EleutherAI/pythia-2.8b-deduped --target-model {os.path.join(compressed_models, task, step)} --delta subtract --input-file {test_datafile} --input-field input --max-length 32 --output-file {os.path.join(output_dir, task, step)}.jsonl"
                jobs.append(job)

for job in jobs:
    os.system(f"ts --gpus 1 {job}")
    # print(job)
    # break