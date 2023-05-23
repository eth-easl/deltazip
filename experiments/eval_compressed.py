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
delta_path = '.cache/compressed_models/opt-1.3b'

os.system(f"ts -S 4")
for task in tasks:
    for delta in [x for x in os.listdir(delta_path) if x.find(task) != -1]:
    # first evaluate base_model
        job = f"python cli/ni_delta_main.py --base-model {base_model} --task {task} --delta-path {delta_path}/{delta}"
        os.system(f"ts --gpus 1 {job}")
