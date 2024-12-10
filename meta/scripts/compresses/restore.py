import os
base_models = ['google.gemma-2-9b-it']
configs = ['2b_2n4m_128bs']

for model in base_models:
    for config in configs:
        hf_name = f'deltazip/{model}.{config}'
        job = f"python cli/restore.py --target-model {hf_name} --outdir .local/merged_models"
        os.system(job)