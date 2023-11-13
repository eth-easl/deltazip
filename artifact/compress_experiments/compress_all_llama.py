import os

cache_folder = os.environ.get("YAO_CACHE")
bits = 4
sparsity = 0
in_folder = os.path.join(
    cache_folder, "experiments", "fmzip", "finetuned_raw", "llama-3b"
)

poi_tasks = [
    'task151_tomqa_find_location_easy_clean',
    'task152_tomqa_find_location_easy_noise',
    'task523_find_if_numbers_or_alphabets_are_more_in_list', 
    'task936_defeasible_nli_snli_classification', 
    'task380_boolq_yes_no_question', 
    'task1308_amazonreview_category_classification', 
    'task515_senteval_odd_word_out', 
    'task372_synthetic_palindrome_numbers', 
    'task227_clariq_classification'
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
for task in tasks:
    if task in poi_tasks:
        steps = os.listdir(os.path.join(in_folder, task))
        for step in steps:
            if not os.path.exists(os.path.join(out_dir, task, step, "config.json")):
                job = f"python cli/compress.py --target-model {os.path.join(in_folder, task, step)} --outdir {os.path.join(out_dir, task, step)} --dataset {os.path.join(ar_dataset, task+'.train.jsonl')} --n-samples 512 --bits {bits} --group-size 128 --sparsity {sparsity} --lossless gdeflate --delta subtract --base-model openlm-research/open_llama_3b_v2"
                jobs.append(job)

os.system("ts -S 3")
for job in jobs:
    os.system(f"ts --gpus 1 {job}")
