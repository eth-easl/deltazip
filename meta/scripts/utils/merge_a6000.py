import os
ROOT_PATH = ".local/compressed_models"
command_format = """
ts -G 1 python cli/merge.py --base-model meta-llama/Llama-2-13b-hf --target-model .local/compressed_models/<model> --output-dir .local/merged_models/<model>
"""

merged_models = [x for x in os.listdir(ROOT_PATH) if "7b" in x]
for model in merged_models:
    new_job = command_format.replace("<model>", model)
    os.system(new_job)