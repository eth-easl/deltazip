import os
from tqdm import tqdm

SRC = ".idea/models/"
DST = "/scratch/xiayao/models/"

os.makedirs(DST, exist_ok=True)

NUM_MODELS = 8
PREFIXS = ["lmsys.vicuna-7b-v1.5.2b50s", "lmsys.vicuna-7b-v1.5.2b50s-tp_2"]

for prefix in PREFIXS:
    for i in tqdm(range(1, NUM_MODELS + 1)):
        src = f"{SRC}{prefix}-{i}"
        dst = f"{DST}{prefix}-{i}"
        # check if src exists
        print(f"cp -r {src} {dst}")
        if not os.path.exists(src):
            print(f"{src} does not exist")
            continue
        # check if dst exists
        if os.path.exists(dst):
            continue
        os.system(f"cp -r {src} {dst}")
