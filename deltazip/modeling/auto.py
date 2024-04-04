from typing import Optional, Union

from ._base import BaseCompressionConfig, BaseDeltaZipModelForCausalLM
from ._utils import check_and_get_model_type
from .bloom import BloomDeltaZipForCausalLM
from .gpt_neox import GPTNeoXDeltaZipForCausalLM
from .gptj import GPTJDeltaZipForCausalLM
from .gpt2 import GPT2DeltaZipForCausalLM
from .llama import LlamaDeltaZipForCausalLM
from .moss import MOSSDeltaZipForCausalLM
from .opt import OPTDeltaZipForCausalLM
from inspect import signature

DeltaZip_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomDeltaZipForCausalLM,
    "gpt_neox": GPTNeoXDeltaZipForCausalLM,
    "gptj": GPTJDeltaZipForCausalLM,
    "gpt2": GPT2DeltaZipForCausalLM,
    "llama": LlamaDeltaZipForCausalLM,
    "opt": OPTDeltaZipForCausalLM,
    "moss": MOSSDeltaZipForCausalLM,
}


class AutoDeltaZipModelForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoDeltaZipModelForCausalLM is designed to be instantiated\n"
            "using `AutoDeltaZipModelForCausalLM.from_pretrained` if want to compress a pretrained model.\n"
            "using `AutoDeltaZipModelForCausalLM.from_compressed` if want to inference with compressed model."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        compress_config: BaseCompressionConfig,
        max_memory: Optional[dict] = None,
        device_map: Optional[str] = None,
        **model_init_kwargs
    ) -> BaseDeltaZipModelForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path)
        return DeltaZip_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            compress_config=compress_config,
            max_memory=max_memory,
            device_map=device_map,
            **model_init_kwargs
        )

    @classmethod
    def from_lora(
        cls,
        pretrained_model_name_or_path: str,
        compress_config: BaseCompressionConfig,
        max_memory: Optional[dict] = None,
        **model_init_kwargs
    ) -> BaseDeltaZipModelForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path)
        return DeltaZip_CAUSAL_LM_MODEL_MAP[model_type].from_lora(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            compress_config=compress_config,
            max_memory=max_memory,
            **model_init_kwargs
        )

    @classmethod
    def from_compressed(
        cls,
        save_dir: str,
        device_map: Optional[str] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        strict: bool = True,
        use_triton: bool = False,
        inject_fused_attention: bool = False,
        inject_fused_mlp: bool = False,
        use_cuda_fp16: bool = True,
        use_bfloat16: bool = False,
        compress_config: Optional[BaseCompressionConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = True,
        unpack: bool = False,
        use_exllama: bool = False,
        **kwargs
    ) -> BaseDeltaZipModelForCausalLM:
        model_type = check_and_get_model_type(save_dir)
        decompress_func = DeltaZip_CAUSAL_LM_MODEL_MAP[model_type].from_compressed
        keywords = {
            key: kwargs[key]
            for key in signature(decompress_func).parameters
            if key in kwargs
        }
        return decompress_func(
            save_dir=save_dir,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            strict=strict,
            use_triton=use_triton,
            inject_fused_attention=inject_fused_attention,
            inject_fused_mlp=inject_fused_mlp,
            use_cuda_fp16=use_cuda_fp16,
            compress_config=compress_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            unpack=unpack,
            use_exllama=use_exllama,
            **keywords
        )


__all__ = ["AutoDeltaZipModelForCausalLM"]
