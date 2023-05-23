import os

tasks = [
    'answer_verification',
    'coherence_classification',
    'dialogue_state_tracking',
    'fact_verification',
    'gender_classification',
    'irony_detection',
    'toxic_language_detection',
    'word_semantics'
]
base_model = 'facebook/opt-1.3b'
os.system(f"ts -S 4")
for task in tasks:
    os.system(f"ts --gpus 1 {job}")
    # then evaluate the finetuned models
    finetuned_model = f".cache/models/opt-1.3b/{task}"
    
    job = f"python cli/ni_eval_gptq.py --base-model {finetuned_model} --task {task}"
    os.system(f"ts --gpus 1 {job}")
