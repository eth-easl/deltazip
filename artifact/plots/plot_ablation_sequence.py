import os
import json
import pandas as pd
from artifact.plots.utils import get_provider_name, get_provider_order

bsz = 2
tokens = [64, 128, 256, 512, 1024, 2048]
plot_data = []

for token in tokens:
    with open(f"artifact/results/ablation/sequence/bsz=2/", "r") as fp:
        results = json.load(fp)
    # calculate 
    for item in results:
        provider = get_provider_name(item['system'])
        provider = get_provider_name(provider)
        total_jobs = len(item["results"])
        total_time_elapsed = max(item["results"], key=lambda x: x["time_elapsed"])["time_elapsed"]