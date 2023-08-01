from safetensors.torch import safe_open

with safe_open(".cache/compressed_models/opt-125m-QA-squad/fmzip-compressed.safetensors", framework='torch') as fp:
    keys = fp.keys()
    metadata = fp.metadata()
    print(metadata['shape'])