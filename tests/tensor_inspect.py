from safetensors.torch import safe_open

with safe_open(
        ".cache/compressed_models/p2.8b_gsd_133/fmzip-compressed.safetensors", 
        framework='torch'
    ) as fp:
    keys = fp.keys()
    metadata = fp.metadata()
    print(metadata['shape'])