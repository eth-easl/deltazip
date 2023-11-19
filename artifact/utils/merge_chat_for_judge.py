import json

# format:
# {"question": "", "answers": [{"from": "fmzip", "text": ""},{"from": "orginal", "text": ""}]}

with open("artifact/results/lmsys_output_vicuna_16bit.jsonl", "r") as fp:
    full_data = [json.loads(line) for line in fp][:1000]

with open("artifact/results/lmsys_output_vicuna_2bit.jsonl", "r") as fp:
    fmzip_data = [json.loads(line) for line in fp][:1000]
data = []

for full, fmzip in zip(full_data, fmzip_data):
    question = full["text"].split("\n")[0].replace("USER: ", "")
    answers = []
    answers.append({"from": "original", "text": full["prediction"][0].strip()})
    answers.append({"from": "fmzip", "text": fmzip["prediction"][0].strip()})
    datum = {"question": question, "answers": answers}
    data.append(datum)

with open("artifact/results/vicuna_agg.jsonl", "w") as fp:
    for d in data:
        fp.write(json.dumps(d) + "\n")
