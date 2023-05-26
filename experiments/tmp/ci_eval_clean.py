import os
import json
import numpy as np
import scipy.stats

tasks = os.listdir(".cache/eval_stats")
confidence = 0.95
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)

for task in tasks:
    evals = os.listdir(os.path.join(".cache/eval_stats", task))
    for e in evals:
        with open(os.path.join(".cache/eval_stats", task, e)) as fin:
            data = json.load(fin)
            if 'exact_match_lower_bound' in data:
                data.pop('exact_match_lower_bound')
                data.pop('exact_match_upper_bound')
                with open(os.path.join(".cache/eval_stats", task, e), 'w') as fout:
                    json.dump(data, fout, indent=2)
