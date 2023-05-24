import os
import json
import pandas as pd
model_size = 'opt-1.3b'
compressed_model_name_prefix = 'facebook.opt-1.3b-.cache.compressed_models.opt-1.3b.'
uncompressed_model_name_prefix = '.cache.models.opt-1.3b.'
base_model_name_prefix='facebook.opt-1.3b'
tasks = os.listdir(f'.cache/eval_stats/')

data = []
"""
data is a list of dicts with the following keys:
- task: the eval task
- finetuned_on: the task where the model is finetuned on
- method: base, uncompressed, fmzip, gptq
- wbit: for fmzip and gptq, the number of bits used
- sparsity: for fmzip, the sparsity used
"""


for task in tasks:
    results = [x for x in os.listdir(f'.cache/eval_stats/{task}/') if x.find(model_size) != -1]
    for result in results:
        method = ''
        with open(f'.cache/eval_stats/{task}/{result}') as f:
            eval_res = json.load(f)
            if result.startswith(compressed_model_name_prefix):
                model_name = result.replace(compressed_model_name_prefix, '')
                finetuned_on = model_name.split('-')[0]
                if 'delta' in model_name:
                    method = 'fmzip'
                    wbit = model_name.split('-')[1].replace("bit", "")
                    sparsity = model_name.split('-')[3].replace("s", "")
                else:
                    method = 'gptq'
                    wbit = model_name.split('-')[1].replace("bit", "")
                    sparsity = 'None'
            elif result.startswith(uncompressed_model_name_prefix):
                method = 'uncompressed'
                model_name = result.replace(uncompressed_model_name_prefix, '')
                finetuned_on = model_name.replace('_stats.json', '')
                wbit = 'None'
                sparsity = 'None'
            elif result.startswith(base_model_name_prefix):
                method = 'base'
                model_name = result.replace(base_model_name_prefix, '')
                finetuned_on = 'None'
                wbit = 'None'
                sparsity = 'None'
            else:
                raise Exception(f'Unknown model name: {result}')
        data.append({
            'task': task,
            'finetuned_on': finetuned_on,
            'method': method,
            'wbit': wbit,
            'sparsity': sparsity,
            'em_acc': eval_res['exact_match_default_track'],
        })

df = pd.DataFrame(data)
df.to_csv('results/eval_results.csv', index=False)