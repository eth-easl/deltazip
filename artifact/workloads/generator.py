import os
import json
import argparse
import datasets
import pandas as pd

to_eval_models = [
    "lmsys/vicuna-7b-v1.5",
    "Xwin-LM/Xwin-LM-7B-V0.1",
    "migtissera/Synthia-7B-v1.2",
    "meta-llama/Llama-2-7b-chat-hf",
    "FlagAlpha/Llama2-Chinese-7b-Chat",
]


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
    trace = trace.shuffle().select(range(args.num_queries))
    for item in trace:
        traces_data.append(
            {
                "prompt": "USER: "
                + item["conversation_a"][0]["content"]
                + "\nASSISTANT: ",
                "timestamp": (item["tstamp"] - min_tstamp) / 10000,
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
