import os
import json
import uuid
import argparse
from scripts.helpers.common import run, get_sys_info
import time

def before_benchmark(args):
    with open(args.workload, "r") as f:
        workload = [json.loads(line) for line in f]
    # translate "base-model" to the actual model name
    warmup = args.warmup_strategy
    system_ready = False
    # wait until system is ready
    while not system_ready:
        try:
            sysinfo = get_sys_info(args.endpoints[0])
            system_ready = True
        except Exception as e:
            print(f"Waiting for 10 secs for the system to be ready: {e}", flush=True)
            time.sleep(10)
    print(f"Translating from base-model to {sysinfo['model']}", flush=True)
    for job_id, job in enumerate(workload):
        if job["model"] == "base-model":
            workload[job_id]["model"] = sysinfo["model"]
    return args.endpoints, workload, warmup, sysinfo


def generate_annotation(endpoints, sysinfo, workload):
    tp_degree = sysinfo["tensor_parallel_size"]
    rp_degree = len(endpoints)
    annotations = f"{workload},tp_degree={tp_degree},rp_degree={rp_degree}"
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="workload.json")
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument(
        "--warmup-strategy", type=str, default="random", choices=["random", "none"]
    )
    parser.add_argument("--endpoints", default=["http://localhost:8000"], nargs="+")
    parser.add_argument("--output", type=str, default="outputs/")
    args = parser.parse_args()

    endpoints, workload, warmup, sysinfo = before_benchmark(args)
    workload_annotation = args.workload.split("/")[-1].split(".")[0]
    annotations = generate_annotation(args.endpoints, sysinfo, workload_annotation)

    outputs = run(endpoints, workload, warmup, sysinfo["model"], sysinfo)
    new_unique_name = str(uuid.uuid4())
    output_file = os.path.join(args.output, f"{new_unique_name}.jsonl")

    with open(output_file, "w") as f:
        meta = {
            "workload": args.workload,
            "endpoints": args.endpoints,
            "warmup_strategy": args.warmup_strategy,
            "annotations": annotations,
            "sys_info": sysinfo,
        }
        f.write(json.dumps(meta))
        f.write("\n")
        for output in outputs:
            f.write(json.dumps(output))
            f.write("\n")
    print(f"Results written to {output_file}", flush=True)

    pid = sysinfo["pid"]
    print(f"Killing process {pid}...")
    os.system(f"kill -9 {pid}")
"""
Example: 
python bench.py --workload .artifact/workloads/distribution=uniform,ar=3.0,duration=30.0.jsonl --base-model meta-llama/Llama-2-7b-hf --output .artifact/benchmarks/results
"""