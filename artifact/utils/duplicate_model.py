import shutil
num_duplication = 10
from_model = ".cache/raw_models/openllama-3b-chat-lossless"

for i in range(num_duplication):
    shutil.copytree(
        from_model,
        f".cache/raw_models/openllama-3b-chat-lossless-{i}",
    )