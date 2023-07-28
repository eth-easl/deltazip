import os
from loguru import logger
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM
import pandas as pd
import random
from fmzip.lossless.service import CompressedInferenceService

base_model_name = "EleutherAI/pythia-2.8b-deduped"
base_branch = "120000"
api = HfApi()

branches = [x.name for x in api.list_repo_refs(base_model_name).branches]
branches = branches[1:]
# remove "step" in front of the branch name
branches = [x.replace("step", "") for x in branches if x.startswith('step')]
branches = sorted(branches, key=lambda x: int(x))
branches = [x for x in branches if int(x) > int(base_branch)]
# shuffle branches randomly
# random.shuffle(branches)

service = CompressedInferenceService(
    base_model=base_model_name,
    dtype='fp16',
    revision=f'step{base_branch}'
)
results = []
for branch in branches:
    print(f"processing branch {branch}")
    dest = os.path.join(".cache", "compressed_model", base_model_name.replace("/", "_"), branch)
    os.makedirs(dest, exist_ok=True)
    compression_rate, time_spent = service.compress_model(
        target_model=base_model_name,
        dest=dest,
        low_gpu_mem=True,
        delta=True,
        revision=f"step{branch}"
    )
    logger.info("Compression rate: {}x".format(compression_rate))
    logger.info("Time spent: {}s".format(time_spent))
    results.append({
        'branch': branch,
        'compression_rate': compression_rate,
        'time_spent': time_spent
    })

    df = pd.DataFrame(results)
    df.to_csv('.cache/compression_rate.csv', index=False)