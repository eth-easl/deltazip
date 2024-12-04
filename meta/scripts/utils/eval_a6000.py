import os
ROOT_PATH = ".local/merged_models"
command_format = """
ts -G 1 lm_eval \
    --use_cache .local/cache/<model> \
    --model_args pretrained=.local/merged_models/<model>,dtype=float16 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa,gsm8k,truthfulqa \
    --output_path .local/eval_results/<model> \
    --batch_size 4
"""


merged_models = [x for x in os.listdir(ROOT_PATH) if "13b" in x]

print(merged_models)
for model in merged_models:
    new_job = command_format.replace("<model>", model)
    os.system(new_job)