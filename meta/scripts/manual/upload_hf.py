import os
from huggingface_hub import create_repo
compressed_models = os.listdir('.local/compressed_models')

for i, cm in enumerate(compressed_models):
    print(f"{i+1}/{len(compressed_models)}: {cm}")
    create_repo(f"espressor/{cm}", repo_type="model", private=True)
    os.system(f"cd .local/compressed_models/{cm} && git init && git remote add origin git@hf.co:espressor/{cm} && git checkout -b main && git lfs track *.safetensors && git add * && git commit -m 'Initial commit' && git push -u origin main -f")