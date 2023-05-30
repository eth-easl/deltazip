import os

model_dir = "/mnt/ds3lab-scratch/xiayao/cache/finetuned_models/opt-1.3b/hf"
dataset_dir = "/mnt/ds3lab-scratch/xiayao/cache/ft_data/natural-instructions"
compressed_model_dir = "/mnt/ds3lab-scratch/xiayao/cache/compressed_models"
tasks = os.listdir(model_dir)

model_size = 'opt-1.3b'

wbits = [2,3,4]
sparsities = [0, 0.9, 0.95, 0.99]
groupsizes = [128, 1024]
jobs = []

for task in tasks:
    for wbit in wbits:
        for sparsity in sparsities:
            for groupsize in groupsizes:
                job = f"python cli/delta_preset.py --base-model facebook/{model_size} --target-model {model_dir}/{task} --dataset {dataset_dir}/{task}.jsonl --wbit {wbit} --sparsity {sparsity} --group-size {groupsize} --out-dir {compressed_model_dir}/{model_size} --n-samples 128"
                jobs.append(job)

os.system(f"ts -S 4")

for job in jobs:
    os.system(f"ts --gpus 1 {job}")