from safetensors.torch import load_file

file_path = ".cache/model_delta.safetensors"
loaded = load_file(file_path)

print(loaded.keys())