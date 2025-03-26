import os
import argparse
import pandas as pd
from vllm.tools.utils import parse_data, get_short_system_name


def peek_perf(args):
    # if it is a folder, then parse all files
    if os.path.isdir(args.input):
        inputs = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".jsonl")
        ]
    else:
        inputs = [args.input]
    for input_file in inputs:
        metadata, results = parse_data(input_file)
        short_sys_name = get_short_system_name(metadata)
        print(f"--- {short_sys_name} ---")
        results = pd.DataFrame(results)
        types = results["type"].unique()
        for type in types:
            # average over all runs
            type_results = results[results["type"] == type]
            mean = type_results["time"].mean()
            max = type_results["time"].max()
            median = type_results["time"].median()
            print(f"{type}    \t{mean:.2f} - {max:.2f}, {median:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    peek_perf(args)
