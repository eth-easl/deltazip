import os

tasks = [
    'answer_verification',
    #'coherence_classification',
    # 'dialogue_state_tracking',
    # 'fact_verification',
    #'gender_classification',
    'irony_detection',
    'toxic_language_detection',
    'word_semantics'
]
base_model = 'facebook/opt-1.3b'
os.system(f"ts -S 4")

for task1 in tasks:
    finetuned_model = f".cache/models/opt-1.3b/{task1}"
    for task2 in tasks:
        if task1 != task2:
            job = f"python cli/ni_main.py --base-model {finetuned_model} --task {task2}"
            os.system(f"ts --gpus 1 {job}")