import os

supported_base_models = [
    'openlm-research/open_llama_3b_v2',
    'EleutherAI/pythia-2.8b-deduped',
]
cache_folder = os.environ.get("YAO_CACHE")
ar_dataset = os.path.join(cache_folder, "datasets", "qi", "ar")
poi_tasks = [
    "task151_tomqa_find_location_easy_clean",
    "task152_tomqa_find_location_easy_noise",
    "task372_synthetic_palindrome_numbers",
    "task227_clariq_classification",
]
poi_steps = {
    'openlm-research/open_llama_3b_v2': [140, 160, 105, 75],
    'EleutherAI/pythia-2.8b-deduped': [110, 72, 120, 44],
}
PRINT_JOB = True
OUTPUT_DIR = os.path.join(cache_folder, "experiments", "fmzip", "compressed_models_new")
bits = [2, 4]
sparsity = [0, 0.5]
def render_job(
        is_delta: False,
        base_model: str,
        task: str,
        step: str,
        bits: int,
        sparsity: float,
        n_samples: int=1024,
        block_size: int = 128,
    ):
    model_dir = os.path.join(cache_folder, "experiments", 'fmzip', 'finetuned_raw', base_model.split('/')[-1])
    target_model_dir = os.path.join(model_dir, task, f'global_step{step}')
    dataset_file =os.path.join(ar_dataset, task+".train.jsonl")
    if is_delta:
        output_dir = os.path.join(OUTPUT_DIR, f"{base_model.split('/')[-1]}-{bits}b{sparsity}s_fmzip", task, f'global_step{step}')
    else:
        output_dir = os.path.join(OUTPUT_DIR, f"{base_model.split('/')[-1]}-{bits}b{sparsity}s_sparsegpt", task, f'global_step{step}')
    if not os.path.exists(output_dir):
        job = f"python cli/compress_v2.py --base-model {base_model} --target-model {target_model_dir} --dataset {dataset_file} --bits {bits} --sparsity {sparsity} --outdir {output_dir} {'--delta subtract' if is_delta else ''} --lossless gdeflate --n-samples {n_samples} --block-size {block_size}"
    return job

if __name__=="__main__":
    os.system("ts -S 1")
    jobs = []
    for base_model in supported_base_models:
        for task in poi_tasks:
            for step in poi_steps[base_model]:
                for bit in bits:
                    for sp in sparsity:
                        jobs.append(render_job(False, base_model, task, str(step), bit, sp))
                        jobs.append(render_job(True, base_model, task, str(step), bit, sp))
    if PRINT_JOB:
        for job in jobs:
            print(job)
    for job in jobs:
        os.system(f"ts --gpus 1 {job}")