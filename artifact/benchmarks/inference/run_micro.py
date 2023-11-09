import os
import argparse
import json


def run(args):
    systems = [
        x
        for x in os.listdir(args.workload)
        if x.endswith(".json") and x.startswith("system")
    ]
    jobs = [
        x
        for x in os.listdir(args.workload)
        if x.endswith(".json") and x.startswith("job")
    ]
    sys_configs = []
    for config in systems:
        with open(os.path.join(args.workload, config), "r") as fp:
            system = json.load(fp)
            sys_configs.append((system, config))
    sys_configs = sorted(sys_configs, key=lambda x: x[0]["systems"][0]["order"])

    for config in sys_configs:
        system = os.path.join(args.workload, config[1])
        for job in jobs:
            job = os.path.join(args.workload, job)
            job = f"python artifact/benchmarks/inference/throughput.py --systems {system} --output artifact/results/local.json --jobs {job}"
            os.system(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    run(args)
