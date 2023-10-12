import json
import torch
import argparse
from copy import deepcopy
from timeit import default_timer as timer
from dataclasses import dataclass
from fmzip.rest.inference import InferenceService

inference_service = None

@dataclass
class Query:
    model: str
    prompt: str
    timestamp: float

def main(args):
    print(args)
    with open(args.workload, "r") as fp:
        workload = json.load(fp)
    backends = workload['systems']
    model_mapping = workload['compressed_model_mapping']
    benchmark_results = []
    for backend in backends:
        inference_service = InferenceService(
            provider = backend['name'],
            model_parallel_strategy=backend['args'].get('model_parallel_strategy', 'none'),
            base_model=workload['base_model'],
            batch_size=backend['args'].get('batch_size', 1),
            max_num_deltas=backend['args'].get('max_num_deltas', 1),
        )
        queries = deepcopy(workload['queries'])
        if backend['mapping']:
            for i, query in enumerate(queries):
                queries[i]['model'] = model_mapping[workload['queries'][i]['model']]
        queries = [Query(**query) for query in queries]
        start = timer()
        response = inference_service.generate(queries=queries, max_new_tokens=512)
        end = timer()
        benchmark_results.append({
            "backend": backend['name'],
            "total_elapsed": end-start,
            "results": response
        })
        del inference_service
        torch.cuda.empty_cache()

    with open(args.output_file, "w") as fp:
        json.dump(benchmark_results, fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="test.json")
    parser.add_argument("--output-file", type=str, default="artifact/results/latency.json")
    args = parser.parse_args()
    main(args)