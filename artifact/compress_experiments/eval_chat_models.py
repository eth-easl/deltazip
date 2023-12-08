import os

cache_dir = os.getenv("YAO_CACHE", "")

eval_models = {
    "lmsys/vicuna-7b-v1.5": os.path.join(
        cache_dir, "experiments/deltazip/compressed_chat_models/vicuna-7b-v1.5-2bits/"
    ),
    "Xwin-LM/Xwin-LM-7B-V0.1": os.path.join(
        cache_dir, "experiments/deltazip/compressed_chat_models/xwin-lm-7b-v0.1-2bits/"
    ),
}


def render_job(
    base_model: str,
    model_name: str,
    is_delta: bool,
    input_file: str,
    input_field: str,
):
    job = f"python cli/lmsys_generate.py --base-model {base_model} --target-model {model_name} --input-file {input_file} --input-field {input_field} --output-file .cache/lmsys_{model_name.replace('/', '-')}_{is_delta}.jsonl --max-length 512 --temperature 0.6 --top-k 50 --top-p 0.9 --fast-tokenizer {'--delta subtract' if is_delta else ''} --n-samples 2048"
    return job


if __name__ == "__main__":
    input_file = "artifact/data/lmsys.jsonl"
    input_field = "text"
    base_model = "meta-llama/Llama-2-7b-hf"
    jobs = []
    for k, v in eval_models.items():
        job = render_job(base_model, k, False, input_file, input_field)
        jobs.append(job)
        job = render_job(base_model, v, True, input_file, input_field)
        jobs.append(job)

    for job in jobs:
        print(job)

    print(f"Total jobs: {len(jobs)}")
    for job in jobs:
        os.system(f"ts --gpus 1 {job}")
