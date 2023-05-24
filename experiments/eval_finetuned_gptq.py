import os

tasks = [
    'answer_verification',
    'coherence_classification',
    # 'dialogue_state_tracking',
    # 'fact_verification',
    'gender_classification',
    'irony_detection',
    'toxic_language_detection',
    'word_semantics'
]
base_model = 'facebook/opt-1.3b'
target_path = '.cache/compressed_models/opt-1.3b'

os.system(f"ts -S 4")
for task in tasks:
    for delta in [x for x in os.listdir(target_path) if x.find(task) == -1 and x.find("delta") == -1]:
    # first evaluate base_model
        job = f"python cli/ni_eval_gptq.py --base-model {base_model} --task {task} --target-model {target_path}/{delta}"
        os.system(f"ts --gpus 1 {job}")