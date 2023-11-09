import os
import json
import argparse
import datasets
import pandas as pd
import numpy as np
from artifact.utils.generators.arrival import PoissonProcess

to_eval_models = [f".cache/raw_models/openllama-3b-chat-{i}" for i in range(1, 19)]
to_eval_models = ["openlm-research/open_llama_3b_v2"] + to_eval_models


def format_openllama(prompt):
    return f"<human>: {prompt}<|endoftext|><assistant>:"


def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"


def get_dialogs():
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    all_dialogs = []
    for idx, item in enumerate(trace):
        all_dialogs.append(format_openllama(item["conversation_a"][0]["content"]))
    return all_dialogs


def pick_model(idx):
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]


def prepare_poisson(args):
    print(args)
    dialogs = get_dialogs()
    poisson_ticks = PoissonProcess(args.arrival_rate).generate_arrivals(
        0, args.duration
    )
    df = pd.read_csv(
        "artifact/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    )
    models = set()
    for row_id, item in df.iterrows():
        models.add(item["func"])
    mapping = {}
    traces_data = []
    for idx, model in enumerate(models):
        mapping[model] = to_eval_models[idx % len(to_eval_models)]
    mapped_models = []
    for row_id, item in df.iterrows():
        mapped_models.append(mapping[item["func"]])
    for idx in range(len(poisson_ticks)):
        traces_data.append(
            {
                "id": idx,
                "prompt": dialogs[idx],
                "timestamp": poisson_ticks[idx],
                "model": mapped_models[idx],
            }
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump({"queries": traces_data}, fp)


def main(args):
    if args.trace == "poisson":
        print("Using Poisson trace")
        prepare_poisson(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="poisson")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=1.5)
    args = parser.parse_args()
    main(args)
