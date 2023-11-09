import os
import argparse
import datasets
import json


def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"


trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
traces_data = []

for item in trace:
    traces_data.append(
        {
            "text": format_lmsys(item["conversation_a"][0]["content"]),
        }
    )

with open("artifact/data/lmsys.jsonl", "w") as fp:
    for line in traces_data:
        fp.write(json.dumps(line) + "\n")
