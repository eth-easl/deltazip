import os

tasks = [
    'answer_verification',
    'coherence_classification',
    'dialogue_state_tracking',
    'fact_verification',
    'gender_classification',
    'irony_detection',
    'toxic_language_detection',
    'word_semantics'
]

model_size = 'opt-1.3b'

wbits = [2,3,4]
sparsities = [0]
jobs = []
for task in tasks:
    for wbit in wbits:
        for sparsity in sparsities:
            job = f"python cli/preset.py --base-model facebook/{model_size} --target-model .cache/models/{model_size}/{task} --dataset .cache/ni_calib/train/{task}.jsonl --wbit {wbit} --sparsity {sparsity} --out-dir .cache/compressed_models/{model_size}"
            jobs.append(job)

os.system(f"ts -S 7")

for job in jobs:
    os.system(f"ts --gpus 1 {job}")