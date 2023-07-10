import os
from tqdm import tqdm
eval_tasks = os.listdir(".cache/eval_tmp/")

for task in eval_tasks:
    references = f".cache/ni_calib/test_references/{task}.jsonl"
    
    eval_results = [x for x in os.listdir(f".cache/eval_tmp/{task}") if x.endswith("_postprocessed.jsonl")]

    os.makedirs(f".cache/eval_tmp_stats/{task}", exist_ok=True)
    
    for eval_result in tqdm(eval_results):
        path_to_eval_result = f".cache/eval_tmp/{task}/{eval_result}"
        output_file = f".cache/eval_tmp_stats/{task}/{eval_result.removesuffix('_postprocessed.jsonl')}_stats.json"
        # if output_file does not exist
        if not os.path.exists(output_file):
            job = f"python cli/ni_eval.py --prediction_file={path_to_eval_result} --reference_file={references} --output_file={output_file}"
            os.system(job)