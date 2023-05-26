import os

base_model = 'facebook/opt-1.3b'
task = 'word_semantics'
delta_paths = [
    '.cache/compressed_models/opt-1.3b/word_semantics-8bit-1024g-0.0s-delta-fulltrainset',
]
# first evaluate base_model
for delta_path in delta_paths:
    job = f"python cli/ni_delta_main.py --base-model {base_model} --task {task} --delta-path {delta_path}"
    os.system(f"ts --gpus 1 {job}")