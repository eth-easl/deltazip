import torch
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from deltazip.utils.delta_utils import subtract_inverse

base_model = "openlm-research/open_llama_3b_v2"
target_model = "/mnt/scratch/xiayao/cache/experiments/deltazip/finetuned_raw/llama-3b/task372_synthetic_palindrome_numbers/global_step105/"
delta_model = "/mnt/scratch/xiayao/cache/experiments/deltazip/compressed_models/4b0s/open_llama_3b_v2/task372_synthetic_palindrome_numbers/global_step105"

compress_config = BaseCompressionConfig(
    bits=4,
    group_size=128,
    sparsity=1,
    prunen=0,
    prunem=0,
    lossless="gdeflate",
    damp_percent=0.02,
)

raw_model = AutoDeltaZipModelForCausalLM.from_pretrained(
    target_model, compress_config=compress_config
)
raw_model = raw_model.half()
raw_model = raw_model.to(torch.device("cuda"))
lm_head = torch.nn.Parameter(raw_model.state_dict()["lm_head.weight"].cuda().half())
embed_token = torch.nn.Parameter(
    raw_model.state_dict()["model.embed_tokens.weight"].cuda().half()
)
del raw_model

with torch.inference_mode():
    print(f"Loading base model...")
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        base_model, compress_config=compress_config
    )
    base_model = base_model.half()
    base_model = base_model.to(torch.device("cuda"))

    print(f"Loading target model...")
    target_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        target_model, compress_config=compress_config
    )
    target_model = target_model.half()
    target_model = target_model.to(torch.device("cuda"))

    print(f"Loading delta model...")
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        delta_model, strict=True, device="cpu", unpack=True
    )
    delta_model = delta_model.half()
    delta_model = delta_model.to(torch.device("cuda"))

    delta_model = subtract_inverse(base_model, delta_model)
    delta_model.lm_head.weight = lm_head
    delta_model.model.embed_tokens.weight = embed_token
    base_model = base_model.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    for name, param in delta_model.state_dict().items():
        print(f"{name}, {torch.max(param- target_model.state_dict()[name])}")
        print(f"{param}")
        print(f"{target_model.state_dict()[name]}")
