from typing import Optional, Union

from ._base import BaseQuantizeConfig, BaseFMZipModelForCausalLM
from ._utils import check_and_get_model_type
from .bloom import BloomFMZipForCausalLM
from .gpt_neox import GPTNeoXFMZipForCausalLM
from .gptj import GPTJFMZipForCausalLM
from .gpt2 import GPT2FMZipForCausalLM
from .llama import LlamaFMZipForCausalLM
from .moss import MOSSFMZipForCausalLM
from .opt import OPTFMZipForCausalLM
from inspect import signature

FMZIP_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomFMZipForCausalLM,
    "gpt_neox": GPTNeoXFMZipForCausalLM,
    "gptj": GPTJFMZipForCausalLM,
    "gpt2": GPT2FMZipForCausalLM,
    "llama": LlamaFMZipForCausalLM,
    "opt": OPTFMZipForCausalLM,
    "moss": MOSSFMZipForCausalLM
}


class AutoFMZipModelForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoFMZipModelForCausalLM is designed to be instantiated\n"
            "using `AutoFMZipModelForCausalLM.from_pretrained` if want to quantize a pretrained model.\n"
            "using `AutoFMZipModelForCausalLM.from_quantized` if want to inference with quantized model."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        max_memory: Optional[dict] = None,
        **model_init_kwargs
    ) -> BaseFMZipModelForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path)
        return FMZIP_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
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
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = True,
        unpack: bool = False,
        **kwargs
    ) -> BaseFMZipModelForCausalLM:
        model_type = check_and_get_model_type(save_dir)
        quant_func = FMZIP_CAUSAL_LM_MODEL_MAP[model_type].from_quantized
        keywords = {key: kwargs[key] for key in signature(quant_func).parameters if key in kwargs}
        return quant_func(
            save_dir=save_dir,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            strict=strict,
            use_triton=use_triton,
            inject_fused_attention=inject_fused_attention,
            inject_fused_mlp=inject_fused_mlp,
            use_cuda_fp16=use_cuda_fp16,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            unpack=unpack,
            **keywords
        )


__all__ = ["AutoFMZipModelForCausalLM"]