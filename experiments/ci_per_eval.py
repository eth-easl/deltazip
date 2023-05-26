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
                pass
            else:
                len_predictions = len(data.keys())-2
                em = 0.01 * data['exact_match_default_track']
                if em == 0:
                    ci_lb = 0
                    ci_ub = 0
                else:
                    ci_length = z_value * np.sqrt((em * (1 - em)) / len_predictions)
                    ci_lb = em - ci_length
                    ci_ub = em + ci_length
                data['exact_match_lower_bound'] = 100 * ci_lb
                data['exact_match_upper_bound'] = 100 * ci_ub
                print(f"task: {task}, eval: {e}, ci_lb: {ci_lb}, ci_ub: {ci_ub}")
                with open(os.path.join(".cache/eval_stats", task, e), 'w') as fout:
                    json.dump(data, fout, indent=2)
