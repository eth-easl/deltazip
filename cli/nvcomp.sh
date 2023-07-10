python benchmarks/nvcomp_delta.py --base-model=facebook/opt-1.3b --model=facebook/opt-iml-max-1.3b --output .cache/model_delta.safetensors
csrc/nvcomp/bin/benchmark -a 2 -f .cache/model_delta.safetensors

python benchmarks/nvcomp_delta.py --model=facebook/opt-iml-max-1.3b --output .cache/model.safetensors
csrc/nvcomp/bin/benchmark -a 2 -f .cache/model.safetensors