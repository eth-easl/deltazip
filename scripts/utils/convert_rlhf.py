import json
import datasets

ds = datasets.load_dataset("Anthropic/hh-rlhf", "en", split="train")
results = []
for datum in ds:
    results.append(
        {
            "text": datum["chosen"],
        }
    )

with open("hh-rlhf.json", "w") as f:
    for result in results:
        f.write(json.dumps(result))
        f.write("\n")
