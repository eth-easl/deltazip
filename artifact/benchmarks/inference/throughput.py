import os
import time
import json
import torch
import subprocess
from timeit import default_timer as timer
from fmzip.utils.randomness import init_seeds
from fmzip.pipelines.fmzip_pipeline import FMZipPipeline
from fmzip.pipelines.hf_pipeline import HuggingFacePipeline

init_seeds(42)

def clear_cache():
    torch.cuda.empty_cache()
    subprocess.check_output(
        "sudo echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True
    )
    torch.cuda.synchronize()

def main(args):
    print(args)
    if os.path.exists(args.output):
        with open(args.output, "r") as fp:
            benchmark_results = json.load(fp)
    else:
        benchmark_results = []
    with open(args.systems, "r") as fp:
        systems = json.load(fp)
    with open(args.jobs, "r") as fp:
        jobs = json.load(fp)
    backends = systems["systems"]
    base_model = jobs["base_model"]
    queries = jobs["queries"]
    mapping = jobs["compressed_model_mapping"]
    gen_configs = jobs["generation_configs"]
    for backend in backends:
        clear_cache()
        if backend["name"] == "hf":
            reformatted_queries = [(x["prompt"], x["model"]) for x in queries]
            pipeline = HuggingFacePipeline(base_model, **backend["args"])
            start = timer()
            results = pipeline.generate(
                reformatted_queries,
                do_sample=True,
                **gen_configs,
            )
            end = timer()
            print(results)
            benchmark_results.append(
                {
                    "backend": backend,
                    "gen_args": gen_configs,
                    "results": results,
                    "total_elapsed": end - start,
                }
            )
        elif backend["name"] == "fmzip":
            reformatted_queries = [
                (
                    x["prompt"],
                    mapping[
                        x["model"]
                        if not backend["args"]["lossless_only"]
                        else x["model"] + "-lossless"
                    ],
                )
                for x in queries
            ]
            pipeline = FMZipPipeline(base_model, **backend["args"])
            start = timer()
            results = pipeline.generate(
                reformatted_queries, do_sample=True, **gen_configs
            )
            end = timer()
            print("results", results)
            benchmark_results.append(
                {
                    "backend": backend,
                    "gen_args": gen_configs,
                    "results": results,
                    "total_elapsed": end - start,
                }
            )
        del pipeline

    with open(args.output, "w") as fp:
        json.dump(benchmark_results, fp, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=str, default="")
    parser.add_argument("--jobs", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    main(args)
