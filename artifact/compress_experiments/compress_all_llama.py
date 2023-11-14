import os

cache_folder = os.environ.get("YAO_CACHE")
bits = 3
sparsity = 0
in_folder = os.path.join(
    cache_folder, "experiments", "fmzip", "finetuned_raw", "llama-3b"
)
include_nodelta = True
include_delta = False
poi_tasks = [
    'task151_tomqa_find_location_easy_clean',
    'task152_tomqa_find_location_easy_noise',
    'task523_find_if_numbers_or_alphabets_are_more_in_list', 
    'task936_defeasible_nli_snli_classification', 
    'task380_boolq_yes_no_question', 
    'task1308_amazonreview_category_classification', 
    'task372_synthetic_palindrome_numbers', 
    'task227_clariq_classification'
]
poi_steps = [
    '140',
    '160',
    '120',
    '120',
    '270',
    '150',
    '105',
    '75',
]
out_dir = os.path.join(
    cache_folder,
    "experiments",
    "fmzip",
    "compressed_models",
    f"{bits}b{sparsity}s",
    "open_llama_3b_v2",
)
ar_dataset = os.path.join(cache_folder, "datasets", "qi", "ar")
tasks = os.listdir(in_folder)

jobs = []
for i, task in enumerate(poi_tasks):
    steps = os.listdir(os.path.join(in_folder, task))
    for step in steps:
        if poi_steps[i] in step:
            delta_out_dir = os.path.join(out_dir, task, step)
            nodelta_out_dir = os.path.join(out_dir+'_nodelta', task, step)
            if include_delta:
                if not os.path.exists(os.path.join(delta_out_dir, "config.json")):
                    job = f"python cli/compress.py --target-model {os.path.join(in_folder, task, step)} --outdir {os.path.join(out_dir, task, step)} --dataset {os.path.join(ar_dataset, task+'.train.jsonl')} --n-samples 512 --bits {bits} --group-size 128 --sparsity {sparsity} --lossless gdeflate --delta subtract --base-model openlm-research/open_llama_3b_v2"
                    jobs.append(job)
            if include_nodelta:
                if not os.path.exists(os.path.join(nodelta_out_dir, "config.json")):
                    job = f"python cli/compress.py --target-model {os.path.join(in_folder, task, step)} --outdir {nodelta_out_dir} --dataset {os.path.join(ar_dataset, task+'.train.jsonl')} --n-samples 512 --bits {bits} --group-size 128 --sparsity {sparsity} --lossless gdeflate --base-model openlm-research/open_llama_3b_v2"
                    jobs.append(job)

os.system("ts -S 4")
for job in jobs:
    os.system(f"ts --gpus 1 {job}")
