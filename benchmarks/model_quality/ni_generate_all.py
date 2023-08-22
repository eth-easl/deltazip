import os

CACHE_PATH = os.environ.get("YAO_CACHE", "")

compressed_models = os.path.join(CACHE_PATH, "experiments/fmzip/compressed_models/auto/pythia-2.8b-deduped")

data_dir = os.path.join(CACHE_PATH, "datasets/qi/test")

tasks = os.listdir(compressed_models)
output_dir = os.path.join(CACHE_PATH, "experiments/fmzip/generation_results")
jobs = []

for task in tasks:
    steps = os.listdir(os.path.join(compressed_models, task))
    for step in steps:
        output_file = os.path.join(output_dir, task, step)
        os.makedirs(output_file, exist_ok=True)
        test_datafile = os.path.join(data_dir, task+".test.jsonl")
        job = f"python cli/ni_generate.py --base-model EleutherAI/pythia-2.8b-deduped --target-model {os.path.join(compressed_models, task, step)} --delta subtract --input-file {test_datafile} --input-field input --do-sample --top-p 0.9 --top-k 0 --temperature 0.9 --max-length 64 --output-file {os.path.join(CACHE_PATH, 'experiments/fmzip/generation_results', task, step)}.jsonl"
        print(job)