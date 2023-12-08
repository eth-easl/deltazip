import os

supported_base_models = [
    "openlm-research/open_llama_3b_v2",
    "EleutherAI/pythia-2.8b-deduped",
]
cache_folder = os.environ.get("YAO_CACHE")
ar_dataset = os.path.join(cache_folder, "datasets", "qi", "ar")
fast_tokenizer = {"open_llama_3b_v2": False, "pythia-2.8b-deduped": True}
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
OUTPUT_DIR = os.path.join(
    cache_folder, "experiments", "deltazip", "generation_results_new"
)


def render_job(
    is_delta: False,
    base_model: str,
    task: str,
    step: str,
):
    model_dir = os.path.join(
        cache_folder,
        "experiments",
        "deltazip",
        "finetuned_raw",
        base_model.split("/")[-1],
    )
    input_file = os.path.join(
        cache_folder, "datasets", "qi", "test", task + ".test.jsonl"
    )
    target_model_dir = os.path.join(model_dir, task, f"global_step{step}")
    output_dir = os.path.join(
        OUTPUT_DIR,
        f"{base_model.split('/')[-1]}-finetuned",
        task,
        f"global_step{step}.jsonl",
    )
    if not os.path.exists(output_dir) or force:
        job = f"python cli/ni_generate.py --base-model {base_model} --target-model {target_model_dir} --input-file {input_file} --input-field input --max-length 64 --output-file {output_dir} {'--fast-tokenizer' if fast_tokenizer else ''}"
        return job
    else:
        return None


if __name__ == "__main__":
    jobs = []
    for base_model in supported_base_models:
        for i, task in enumerate(poi_tasks):
            step = poi_steps[base_model][i]
            jobs.append(render_job(False, base_model, task, str(step)))

    jobs = [job for job in jobs if job is not None]

    if PRINT_JOB:
        for job in jobs:
            print(job)
    for job in jobs:
        os.system(f"ts --gpus 1 {job}")
