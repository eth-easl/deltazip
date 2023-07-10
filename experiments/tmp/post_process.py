"""
Reads the results of the evaluation, and do some post-processing.
- Since most of the tasks are classification, we only take the text before the first \n
- Save outputs as {path}/{filename}_postprocessed.jsonl
- Then we compare exact match
"""
import os
import json
eval_tasks = os.listdir(".cache/eval_tmp/")

for task in eval_tasks:
    eval_results = os.listdir(f".cache/eval_tmp/{task}")
    for eval_result in [x for x in eval_results if x.find("_postprocessed") == -1]:
        with open(f".cache/eval_tmp/{task}/{eval_result}", "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        for datum in data:
            datum['prediction'] = datum['prediction'].split("\n")[0]

        with open(f".cache/eval_tmp/{task}/{eval_result.removesuffix('.jsonl')}_postprocessed.jsonl", "w") as f:
            for datum in data:
                f.write(json.dumps(datum) + "\n")