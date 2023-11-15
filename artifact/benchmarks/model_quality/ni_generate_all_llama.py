import os

CACHE_PATH = os.environ.get("YAO_CACHE", "")
folder_name = "2b0s"

compressed_models = os.path.join(
    CACHE_PATH, f"experiments/fmzip/compressed_models/{folder_name}/open_llama_3b_v2"
)

data_dir = os.path.join(CACHE_PATH, "datasets/qi/test")

tasks = os.listdir(compressed_models)
output_dir = os.path.join(
    CACHE_PATH, f"experiments/fmzip/generation_llama/new_generation_results_{folder_name}"
)
jobs = []
for task in tasks:
    steps = os.listdir(os.path.join(compressed_models, task))
    for step in steps:
        output_file = os.path.join(output_dir, task)
        os.makedirs(output_file, exist_ok=True)
        test_datafile = os.path.join(data_dir, task + ".test.jsonl")
        if "config.json" in os.listdir(os.path.join(compressed_models, task, step)):
            output_file = f"{os.path.join(output_dir, task, step)}.jsonl"
            if not os.path.exists(output_file):
                job = f"python cli/ni_evaluate.py --base-model openlm-research/open_llama_3b_v2 --target-model {os.path.join(compressed_models, task, step)} --delta subtract --input-file {test_datafile} --input-field input --max-length 32 --output-file {os.path.join(output_dir, task, step)}.jsonl"
                jobs.append(job)

for job in jobs:
    os.system(f"ts --gpus 1 {job}")
    # print(job)
    # break
