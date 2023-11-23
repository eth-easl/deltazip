import os
import json
import argparse
import datasets
import pandas as pd
import numpy as np

def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"


def get_dialogs():
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    all_dialogs = []
    for idx, item in enumerate(trace):
        all_dialogs.append(format_lmsys(item["conversation_a"][0]["content"]))
    return all_dialogs

if __name__=="__main__":
    dialogs = get_dialogs()
    with open("artifact/data/lmsys_dialogs.jsonl", "w") as fp:
        for d in dialogs:
            fp.write(json.dumps({"text": d}) + "\n")