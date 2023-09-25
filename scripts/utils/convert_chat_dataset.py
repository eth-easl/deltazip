import json
import argparse

def convert(format):
    if format=='meta':
        user_identifier = "[INST]"
        bot_identifier = ""
        user_end = "[/INST]"
        bot_end = "</s>"
    elif format == 'lmsys':
        user_identifier = "USER:"
        bot_identifier = "ASSISTANT:"
        user_end = " "
        bot_end = "</s>"
    with open(".cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", "r") as fp:
        data = json.load(fp)
    outputs = []
    for datum in data:
        output_text = ""
        for conv in datum['conversations']:
            if conv['from'] == "human":
                output_text += user_identifier + " " + conv['value'] +user_end
            elif conv['from'] == "gpt":
                output_text += bot_identifier + " " + conv['value'] + bot_end
        outputs.append(output_text)

    with open(f".cache/datasets/{format}.jsonl", "w") as fp:
        for output in outputs:
            fp.write(json.dumps({'text': output}) + "\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, choices=['lmsys', 'meta'], default='lmsys')
    args = parser.parse_args()
    convert(args.format)