from fmzip import AutoFMZipModelForCausalLM, BaseCompressionConfig

model = ".cache/compressed_models/bits-2/llama-2-7b-chat"
delta_model = AutoFMZipModelForCausalLM.from_compressed(
    model, strict=False, device="cpu", low_cpu_mem_usage=False, unpack=True
)
