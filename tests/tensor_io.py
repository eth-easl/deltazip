from safetensors import safe_open

tensors = {}
with safe_open("/experiments/fmzip/finetuned_raw/pythia-2.8b-deduped/sentence_ordering/global_step22/model.safetensors", framework="pt", device=0) as f:
    print(f.keys())
