import os

in_dir = os.path.join(
    os.environ.get("YAO_CACHE"),
    "experiments",
    "fmzip",
    "finetuned_raw",
    "pythia-2.8b-deduped",
)
out_dir = os.path.join(
    os.environ.get("YAO_CACHE"),
    "experiments",
    "fmzip",
    "finetuned_compressed",
    "pythia-2.8b-deduped",
)
dataset_dir = os.path.join(
    os.environ.get("YAO_CACHE"),
    "datasets",
    "quantitative_natural_instructions",
    "train",
)
BASE_MODEL = "EleutherAI/pythia-2.8b-deduped"

os.makedirs(out_dir, exist_ok=True)
delta_ops = ["subtract"]

tasks = os.listdir(in_dir)

sparsities = [0.5]
wbits = [4]
delta_ops = ["subtract"]


def render_job(task, step, sparsity, wbit, delta):
    os.makedirs(os.path.join(out_dir, task, step), exist_ok=True)
    return f"python cli/cli.py --base-model {BASE_MODEL} --delta {delta} --target-model {os.path.join(in_dir, task, step)} --out-dir {os.path.join(out_dir, task, step)} --sparsity {sparsity} --bits {wbit} --dataset {os.path.join(dataset_dir, task+'.train.jsonl')} --lossless gdeflate"


for task in tasks:
    steps = os.listdir(os.path.join(in_dir, task))
    for step in steps:
        for sparsity in sparsities:
            for wbit in wbits:
                for delta_op in delta_ops:
                    job = render_job(task, step, sparsity, wbit, delta_op)
                    print(job)
                    break
