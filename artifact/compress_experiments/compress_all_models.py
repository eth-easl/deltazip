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

def render_job(
        is_nodelta: False,
        base_model: str,
        task: str,
        step: str,
        bits: int,
        sparsity: float
    ):
    model_dir = os.path.join(cache_folder, "experiments", 'fmzip', 'finetuned_raw', base_model.split('/')[-1])
    target_model_dir = os.path.join(model_dir, task, f'global_step{step}')
    