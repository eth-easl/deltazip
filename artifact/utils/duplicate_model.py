import shutil
num_duplication = 10
from_model = ".cache/compressed_models/3b-parameters/openllama-chat-0-lossless"

for i in range(num_duplication):
    shutil.copytree(
        from_model,
        f".cache/compressed_models/3b-parameters/openllama-chat-delta-lossless-{i}",
    )