import os
compressed_models = os.listdir(".local/compressed_models")
for cm in compressed_models:
    job = f"huggingface-cli upload deltazip/{cm} {os.path.join('.local/compressed_models', cm)} --repo-type model"
    print(job)
    os.system(job)