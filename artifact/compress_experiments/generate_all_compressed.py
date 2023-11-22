import os

cache_folder = os.environ.get("YAO_CACHE")
compressed_model_dir = os.path.join(
    cache_folder, "experiments", "fmzip", "compressed_models"
)
fast_tokenizer = {"open_llama_3b_v2": False, "pythia-2.8b-deduped": True}
base_models = [
    "open_llama_3b_v2",
    "pythia-2.8b-deduped",
]
hf_id = {
    "open_llama_3b_v2": "openlm-research/open_llama_3b_v2",
    "pythia-2.8b-deduped": "EleutherAI/pythia-2.8b-deduped",
}
OUTPUT_DIR = os.path.join(
    cache_folder, "experiments", "fmzip", "generation_results_new"
)
FORCE = False
PRINT_JOB = True
keywords = ['2b0.95s']

def render_job(
    base_model, target_model_dir, task, step, is_delta, config, fast_tokenizer
):
    input_file = os.path.join(
        cache_folder, "datasets", "qi", "test", task + ".test.jsonl"
    )
    output_dir = os.path.join(
        OUTPUT_DIR, f"{base_model}", f"{task}-{step}", f"{config}.jsonl"
    )
    job = None
    if not os.path.exists(output_dir) or FORCE or any([keyword in output_dir for keyword in keywords]):
        job = f"python cli/ni_eval.py --base-model {hf_id[base_model]} --target-model {target_model_dir} {'--delta subtract' if is_delta else ''} --input-file {input_file} --input-field input --max-length 64 --output-file {output_dir} {'--fast-tokenizer' if fast_tokenizer else ''}"
    return job


if __name__ == "__main__":
    models = os.listdir(compressed_model_dir)
    jobs = []
    for model in models:
        if "open_llama_3b_v2" in model:
            base_model = "open_llama_3b_v2"
        elif "pythia-2.8b-deduped" in model:
            base_model = "pythia-2.8b-deduped"
        method = model.split("_")[-1]
        config = model.replace(f"{base_model}", "")[1:]
        tasks = os.listdir(os.path.join(compressed_model_dir, model))
        for task in tasks:
            steps = os.listdir(os.path.join(compressed_model_dir, model, task))
            for step in steps:
                job = render_job(
                    base_model=base_model,
                    target_model_dir=os.path.join(
                        compressed_model_dir, model, task, step
                    ),
                    task=task,
                    step=step,
                    is_delta=True if method == "fmzip" else False,
                    config=config,
                    fast_tokenizer=fast_tokenizer[base_model],
                )
                jobs.append(job)

    jobs = [job for job in jobs if job is not None]
    if PRINT_JOB:
        for job in jobs:
            print(job)

    for job in jobs:
        os.system(f"ts --gpus 1 {job}")
