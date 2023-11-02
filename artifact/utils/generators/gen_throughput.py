import os
import json
import argparse
import datasets
import pandas as pd
import numpy as np
from artifact.workloads.generators.arrival import PoissonProcess

# to_eval_models = [
#     "lmsys/vicuna-7b-v1.5",
#     "Xwin-LM/Xwin-LM-7B-V0.1",
#     "migtissera/Synthia-7B-v1.2",
#     "meta-llama/Llama-2-7b-chat-hf",
#     "FlagAlpha/Llama2-Chinese-7b-Chat",
# ]

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


def prepare_lmsys(args):
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    traces_data = []
    models = {}
    tstamps = []
    for item in trace:
        tstamps.append(item["tstamp"])
        if item["model_a"] not in models:
            models[item["model_a"]] = 1
        else:
            models[item["model_a"]] += 1
    min_tstamp = min(tstamps)
    # sort models by number of occurences
    models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    # create a round robin mapping from models to to_eval_models
    mapping = {}
    for idx, model in enumerate(models):
        mapping[model[0]] = to_eval_models[idx % len(to_eval_models)]
    # randomly take num_queries from trace
    # sort trace by timestamp
    trace = sorted(trace, key=lambda x: x["tstamp"])
    # take num_queries from trace, randomly start
    start = np.random.randint(0, len(trace) - args.num_queries)
    trace = trace[start : start + args.num_queries]
    min_tstamp = trace[0]["tstamp"]
    for idx, item in enumerate(trace):
        traces_data.append(
            {
                "id": idx,
                "prompt": format_openllama(item["conversation_a"][0]["content"]),
                "timestamp": (item["tstamp"] - min_tstamp) / 1,
                "model": mapping[item["model_a"]],
            }
        )
    with open(args.output, "w") as fp:
        json.dump({"queries": traces_data}, fp)


def prepare_azure(args):
    df = pd.read_csv("artifact/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    df['tstamp'] = df['end_timestamp'] - df['duration']
    tstamps = []
    models = {}
    for row_id, row in df.iterrows():
        tstamps.append(row["tstamp"])
        if row["func"] not in models:
            models[row["func"]] = 1
        else:
            models[row["func"]] += 1
    min_tstamp = min(tstamps)
    # sort models by number of occurences
    models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    mapping = {}
    for idx, model in enumerate(models):
        mapping[model[0]] = to_eval_models[idx % len(to_eval_models)]
    trace = df.sort_values(by=["tstamp"], ascending=True)
    # take num_queries from trace, randomly start
    start = np.random.randint(0, len(trace) - args.num_queries)
    trace = trace[start : start + args.num_queries]
    min_tstamp = trace.iloc[0]["tstamp"]
    traces_data = []
    dialogs = get_dialogs()
    trace = trace.to_dict('records')
    for idx, item in enumerate(trace):
        traces_data.append(
            {
                "id": idx,
                "prompt": dialogs[idx],
                "timestamp": (item["tstamp"] - min_tstamp) / 1,
                "model": mapping[item["func"]],
            }
        )
    with open(args.output, "w") as fp:
        json.dump({"queries": traces_data}, fp)

def estimate_base_model_prob():
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    traces_data = []
    models = {}
    tstamps = []
    for item in trace:
        tstamps.append(item["tstamp"])
        if item["model_a"] not in models:
            models[item["model_a"]] = 1
        else:
            models[item["model_a"]] += 1
    min_tstamp = min(tstamps)
    # sort models by number of occurences
    models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    # calculate the probability of the first model
    total_count = sum([item[1] for item in models])
    most_popular_model_count = models[0][1]
    return most_popular_model_count / total_count

def random_model_picker(base_model_prob:float):
    model_count = len(to_eval_models)
    other_model_prob = (1 - base_model_prob) / (model_count - 1)
    # now pick a model
    model_probs = [base_model_prob] + [other_model_prob] * (model_count-1)
    return np.random.choice(to_eval_models, p=model_probs)

def prepare_poisson(args):
    dialogs = get_dialogs()
    poisson_ticks = PoissonProcess(args.arrival_rate).generate_arrivals(
        0, args.duration
    )
    traces_data = []
    for idx in range(len(poisson_ticks)):
        traces_data.append(
            {
                "id": idx,
                "prompt": dialogs[idx],
                "timestamp": poisson_ticks[idx],
                "model": random_model_picker(base_model_prob=0.0918)
            }
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump({"queries": traces_data}, fp)


def main(args):
    print(args)
    if args.trace == "lmsys":
        print("Using LMSys trace")
        prepare_lmsys(args)
    elif args.trace == "azure":
        print("Using Azure trace")
        prepare_azure(args)
    elif args.trace == "poisson":
        print("Using Poisson trace")
        prepare_poisson(args)
    elif args.trace == 'estimate':
        prob = estimate_base_model_prob()
        print(prob)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="lmsys")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=1.5)
    args = parser.parse_args()
    main(args)
