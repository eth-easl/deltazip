import os

cache_folder = os.environ.get("YAO_CACHE")

in_folder = os.path.join(cache_folder, "experiments", "fmzip", "finetuned_raw", "pythia-2.8b-deduped")
out_dir = os.path.join(cache_folder, "experiments", "fmzip", "compressed_models","auto" ,"pythia-2.8b-deduped")
ar_dataset = os.path.join(cache_folder, "datasets", "qi", "ar")
tasks = os.listdir(in_folder)

jobs = []
for task in tasks:
    steps = os.listdir(os.path.join(in_folder, task))
    for step in steps:
        job = f"python cli/auto_compress.py --target-model {os.path.join(in_folder, task, step)} --outdir {os.path.join(out_dir, task, step)} --dataset {os.path.join(ar_dataset, task+'.train.jsonl')} --n-samples 256 --lossless gdeflate --delta subtract --base-model EleutherAI/pythia-2.8b-deduped --n-samples 1024 --perc-damp 0.01 --tolerance 1e-10"
        jobs.append(job)
os.system("ts -S 4")
for job in jobs:
    os.system(f"ts --gpus 1 {job}")