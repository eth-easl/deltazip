import os
import json
import argparse
import datasets
import pandas as pd
import numpy as np

# to_eval_models = [
#     "lmsys/vicuna-7b-v1.5",
#     "Xwin-LM/Xwin-LM-7B-V0.1",
#     "migtissera/Synthia-7B-v1.2",
#     "meta-llama/Llama-2-7b-chat-hf",
#     "FlagAlpha/Llama2-Chinese-7b-Chat",
# ]
to_eval_models = [
    "xzyao/openllama-3b-chat",
    "xzyao/openllama-chat-2",
    "xzyao/openllama-chat-3",
]

def format_openllama(prompt):
    return f"<human>: {prompt}<|endoftext|><assistant>:"

def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"

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
    print(mapping)
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
                "prompt":format_openllama(item["conversation_a"][0]["content"]),
                "timestamp": (item["tstamp"] - min_tstamp) / 100,
                "model": mapping[item["model_a"]],
            }
        )
    with open(args.output, "w") as fp:
        json.dump({"queries": traces_data}, fp)


def main(args):
    print(args)
    if args.trace == "lmsys":
        print("Using LMSys trace")
        prepare_lmsys(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="lmsys")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    main(args)
