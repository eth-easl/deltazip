import shutil

num_duplication = 10
from_model = ".cache/compressed_models/3b-parameters/4bits-openllama-0"

for i in range(1, num_duplication):
    shutil.copytree(
        from_model,
        f".cache/compressed_models/3b-parameters/4bits-openllama-{i}",
    )
