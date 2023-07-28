import os

in_dir = os.environ.get("YAO_CACHE", "experiments", "fmzip", "finetuned_raw", "pythia-2.8b-deduped")
out_dir = os.environ.get("YAO_CACHE", "experiments", "fmzip", "finetuned_compressed", "pythia-2.8b-deduped")

BASE_MODEL = 'EleutherAI/pythia-2.8b-deduped'

os.makedirs(out_dir, exist_ok=True)
delta_ops = ['subtract']

tasks = os.listdir(in_dir)

for task in tasks:
    steps = os.listdir(os.path.join(in_dir, task))
    