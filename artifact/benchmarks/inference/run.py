import os
import argparse
def run(args):
    systems = [x for x in os.listdir(args.workload) if x.endswith(".json") and x.startswith("system")]
    jobs = [x for x in os.listdir(args.workload) if x.endswith(".json") and x.startswith("job")]
    for config in systems:
        system = os.path.join(args.workload, config)
        for job in jobs:
            job = os.path.join(args.workload, job)
            job = f"python artifact/benchmarks/inference/throughput.py --systems {system} --output artifact/results/local.json --jobs {job}"
            os.system(job)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    run(args)