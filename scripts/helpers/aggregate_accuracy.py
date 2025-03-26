import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from tabulate import tabulate

def get_dir_size(dir_path):
    return sum(f.stat().st_size for f in Path(dir_path).rglob('*') if f.is_file())

def bytes_to_mb(size):
    return size / (1024 * 1024)

def load_accuracy(accuracy_dir):
    tasks = ['boolq','logiqa']
    metric = 'acc,none'
    accuracies = []
    models = os.listdir(accuracy_dir)
    for model in models:
        eval_results = os.listdir(os.path.join(accuracy_dir, model))
        trial = os.listdir(os.path.join(accuracy_dir, model, eval_results[0]))[0]
        with open(os.path.join(accuracy_dir, model, eval_results[0], trial)) as f:
            data = json.load(f)['results']
            for task in tasks:
                accuracies.append({
                    'model': model,
                    'task': task,
                    'accuracy': f"{100 * data[task][metric]:.2f}%"
                })
            accuracies.append({
                'model': model,
                'task': 'truthfulqa',
                'accuracy': f"{(100*(data['truthfulqa_mc1'][metric] + data['truthfulqa_mc2'][metric])) / 2:.2f}%"
            })
    return accuracies

def main(args):
    print(args)
    original_model_dir = snapshot_download(args.full_model)
    original_size = get_dir_size(original_model_dir)
    compressed_size = get_dir_size(args.compressed_model)
    
    print(f"Original model size: {bytes_to_mb(original_size):.2f}MB\n[DeltaZip] compressed model size: {bytes_to_mb(compressed_size):.2f}MB")
    print(f"[DeltaZip] Compression ratio: {original_size/compressed_size:.2f}x")
    
    if args.sparsegpt_model:
        sparsegpt_size = get_dir_size(args.sparsegpt_model)
        print(f"[SparseGPT] model size: {bytes_to_mb(sparsegpt_size):.2f}MB")
        print(f"[SparseGPT] Compression ratio: {original_size/sparsegpt_size:.2f}x")
        
    if args.awq_model:
        awq_size = get_dir_size(args.awq_model)
        print(f"[AWQ] model size: {bytes_to_mb(awq_size):.2f}MB")
        print(f"[AWQ] Compression ratio: {original_size/awq_size:.2f}x")
    
    accuracies = load_accuracy(args.accuracy_dir)
    table = tabulate(accuracies, headers='keys', tablefmt='pretty')
    print(table)
    
if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate accuracy of multiple models')
    parser.add_argument('--compressed-model', type=str, required=True)
    parser.add_argument('--full-model', type=str, required=True)
    parser.add_argument('--accuracy-dir', type=str, required=True)
    parser.add_argument('--sparsegpt-model', type=str, required=False)
    parser.add_argument('--awq-model', type=str, required=False)
    main(parser.parse_args())