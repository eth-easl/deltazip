import os
import json
import pandas
from glob import glob
from tqdm import tqdm

def parse_model_args(model_args):
    if "deltazip" in model_args:
        model_args = model_args.split("deltazip")[1]
        model_args = model_args.split(",")[0]
        model_args = model_args.strip(".")
        model_args = model_args.replace(".", "/")
    else:
        model_args = model_args.split(",")[0]
        model_args = model_args.replace("pretrained=", "")
    return model_args

def parse_lmeval(args):
    print(args)
    # recursively enumerate all "json" files under args.input

    result_files = [y for x in os.walk(args.input) for y in glob(os.path.join(x[0], '*.json'))]
    print(f"Found {len(result_files)} result files")
    results = []
    for rf in tqdm(result_files):    
        with open(rf, 'r') as f:
            data = json.load(f)
            config = data['config']
            data = data['results']
            for task, eval_res in data.items():
                for eval_key, eval_val in eval_res.items():
                    if eval_key!= 'alias':
                        results.append({
                            'model': parse_model_args(config['model_args']),
                            'task': task,
                            'metric': eval_key.replace(',none', ''),
                            'value': eval_val
                        })
    df = pandas.DataFrame(results)
    df.to_csv('.local/eval_results.csv', index=False)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='.local/eval_results')
    parser.add_argument("--output", type=str, default='.local/eval_results.csv')
    args = parser.parse_args()
    parse_lmeval(args)