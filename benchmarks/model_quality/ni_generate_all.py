import os

CACHE_PATH = os.environ.get("YAO_CACHE", "")

compressed_models = os.path.join(CACHE_PATH, "experiments/fmzip/compressed_models/auto/pythia-2.8b-deduped")

data_dir = os.path.join(CACHE_PATH, "datasets/qi/test")

tasks = os.listdir(compressed_models)
output_dir = os.path.join(CACHE_PATH, "experiments/fmzip/generation_results_0.6")
jobs = []
for task in tasks:
    steps = os.listdir(os.path.join(compressed_models, task))
    for step in steps:
        output_file = os.path.join(output_dir, task)
        os.makedirs(output_file, exist_ok=True)
        test_datafile = os.path.join(data_dir, task+".test.jsonl")
        job = f"python cli/ni_evaluate.py --base-model EleutherAI/pythia-2.8b-deduped --target-model {os.path.join(compressed_models, task, step)} --delta subtract --input-file {test_datafile} --input-field input --max-length 32 --output-file {os.path.join(output_dir, task, step)}.jsonl"
        jobs.append(job)
os.system("TS_VISIBLE_DEVICES=1,2,3 ts -S 4")
for job in jobs:
    os.system(f"TS_VISIBLE_DEVICES=1,2,3 ts --gpus 1 {job}")
    # os.system(f"{job}")
