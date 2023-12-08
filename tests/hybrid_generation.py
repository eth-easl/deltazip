# Mixed Precision Generation
from deltazip.pipelines.deltazip_pipeline import DeltaZipPipeline
from deltazip import AutoDeltaZipModelForCausalLM, BaseCompressionConfig
from deltazip.utils.delta_utils import xor_inverse, subtract_inverse
from transformers import AutoTokenizer
import torch

base_model_name = "meta-llama/Llama-2-7b-hf"
delta = ".cache/compressed_models/7b-parameters/bits-2/vicuna-7b-v1.5"


def load_delta_model():
    compress_config = BaseCompressionConfig(
        bits=4,
        group_size=128,
        sparsity=1,
        prunen=0,
        prunem=0,
        lossless="gdeflate",
        damp_percent=0.02,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(
        base_model_name, compress_config=compress_config
    )
    delta_model = AutoDeltaZipModelForCausalLM.from_compressed(
        delta, strict=False, device="cpu", unpack=True
    )
    delta_model = subtract_inverse(base_model, delta_model)
    del base_model
    torch.cuda.empty_cache()
    delta_model = delta_model.to(torch.device("cuda"))
    return delta_model, tokenizer


def main():
    # delta_model, tokenizer = load_delta_model()
    test_data = [
        ("USER: Who is Alan Turing?\nASSITANT: ", delta),
    ]

    # delta_model_results = []
    # for i in test_data:
    #     output = delta_model.generate(
    #         **tokenizer(i[0], return_tensors="pt").to(delta_model.device),
    #     )
    #     delta_model_results.append(
    #         tokenizer.decode(output[0], skip_special_tokens=True)
    #     )
    # del delta_model
    torch.cuda.empty_cache()
    mpm = DeltaZipPipeline(base_model=base_model_name, placement_strategy="colocate")
    # print("delta model results")
    # print(delta_model_results)
    print("Mixed Precision Model results")
    results = mpm.generate(test_data)
    print(results)


if __name__ == "__main__":
    main()
