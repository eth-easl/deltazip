import os
import argparse
def run(args):
    configs = [x for x in os.listdir(args.workload) if x.endswith(".json")]
    for config in configs:
        workload = os.path.join(args.workload, config)
        os.system(f"python artifact/benchmarks/inference/throughput.py --workload {workload} --output artifact/results/local.json")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    run(args)