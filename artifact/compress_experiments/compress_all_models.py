import os

supported_base_models = [
    "openlm-research/open_llama_3b_v2",
    "EleutherAI/pythia-2.8b-deduped",
]
cache_folder = os.environ.get("YAO_CACHE")
ar_dataset = os.path.join(cache_folder, "datasets", "qi", "ar")

include_fmzip = True
include_sparsegpt = True
force = True
poi_tasks = [
    "task151_tomqa_find_location_easy_clean",
    "task152_tomqa_find_location_easy_noise",
    "task372_synthetic_palindrome_numbers",
    "task227_clariq_classification",
    "task523_find_if_numbers_or_alphabets_are_more_in_list",
    "task936_defeasible_nli_snli_classification",
    "task380_boolq_yes_no_question",
    "task1308_amazonreview_category_classification",
]
poi_steps = {
    "openlm-research/open_llama_3b_v2": [140, 160, 105, 75, 120, 120, 270, 150],
    "EleutherAI/pythia-2.8b-deduped": [198, 96, 108, 88, 144, 171, 342, 161],
}

PRINT_JOB = True
OUTPUT_DIR = os.path.join(cache_folder, "experiments", "fmzip", "compressed_models")
bits = [2, 4]
sparsity = [0, 0.75]


def render_job(
    is_delta: False,
    base_model: str,
    task: str,
    step: str,
    bits: int,
    sparsity: float,
    n_samples: int = 256,
    block_size: int = 128,
):
    model_dir = os.path.join(
        cache_folder, "experiments", "fmzip", "finetuned_raw", base_model.split("/")[-1]
    )
    target_model_dir = os.path.join(model_dir, task, f"global_step{step}")
    dataset_file = os.path.join(ar_dataset, task + ".train.jsonl")
    if is_delta:
        output_dir = os.path.join(
            OUTPUT_DIR,
            f"{base_model.split('/')[-1]}-{bits}b{sparsity}s_fmzip",
            task,
            f"global_step{step}",
        )
    else:
        output_dir = os.path.join(
            OUTPUT_DIR,
            f"{base_model.split('/')[-1]}-{bits}b{sparsity}s_sparsegpt",
            task,
            f"global_step{step}",
        )
    if not os.path.exists(output_dir) or force:
        job = f"python cli/compress.py --base-model {base_model} --target-model {target_model_dir} --dataset {dataset_file} --bits {bits} --sparsity {sparsity} --outdir {output_dir} {'--delta subtract' if is_delta else ''} --lossless gdeflate --n-samples {n_samples} --block-size {block_size} --fast-tokenizer --shuffle-dataset --perc-damp 0.01"
        return job
    else:
        return None


if __name__ == "__main__":
    jobs = []
    for base_model in supported_base_models:
        for i, task in enumerate(poi_tasks):
            step = poi_steps[base_model][i]
            for bit in bits:
                for sp in sparsity:
                    if include_fmzip:
                        jobs.append(
                            render_job(True, base_model, task, str(step), bit, sp)
                        )
                    if include_sparsegpt:
                        jobs.append(
                            render_job(False, base_model, task, str(step), bit, sp)
                        )

    jobs = [job for job in jobs if job is not None]

    if PRINT_JOB:
        for job in jobs:
            print(job)
    os.system("ts -S 1")
    for job in jobs:
        os.system(f"ts --gpus 1 {job}")
