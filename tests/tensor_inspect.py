from safetensors.torch import safe_open

with safe_open(
    ".cache/compressed_models/llama-2-7b-chat/fmzip-compressed.safetensors",
    framework="torch",
) as fp:
    keys = fp.keys()
    metadata = fp.metadata()
    print(metadata["shape"])
