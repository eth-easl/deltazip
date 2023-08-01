import os
import copy
import json
import torch
import cupy as cp
import accelerate
import transformers
import torch.nn as nn
from loguru import logger
from os.path import join, isfile
from safetensors.numpy import safe_open
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field, fields
from transformers.utils.hub import PushToHubMixin
from safetensors.numpy import save_file as safe_save
from accelerate.hooks import remove_hook_from_module
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers


from ._const import *
from ._utils import (
    pack_model,
    get_module_by_name,
    find_layers,
    move_to_device,
    get_device,
    make_quant,
    unpack_model,
)
from ..nn_modules._fused_base import FusedBaseAttentionModule, FusedBaseMLPModule
from ..core.gptq import GPTQ
from ..core.quant import Quantizer
from ..core.sparsegpt import SparseGPT
from ..utils.data_utils import collate_data
from ..lossless.compressor import LosslessCompressor


@dataclass
class BaseCompressionConfig(PushToHubMixin):
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    # sparsity = how many parameters we set to zero after quantization
    sparsity: float = field(default=0)
    prunen: int = field(default=2)
    prunem: int = field(default=4)
    group_size: int = field(default=-1)
    block_size: int = field(default=1)
    group_rows: int = field(default=1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    lossless: str = field(default="none")
    dtype: str = field(default="fp16")

    def __post_init__(self):
        fields_info = fields(self)
        if self.sparsity < 0 or self.sparsity > 1:
            raise ValueError(f"sparsity must be [0, 1]")
        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(
                f"only support quantize to {fields_info[0].metadata['choices']} bits."
            )
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("unless equal to -1, group_size must greater then 0.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, "compress_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str):
        with open(join(save_dir, "compress_config.json"), "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "group_rows": self.group_rows,
            "sparsity": self.sparsity,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "lossless": self.lossless,
        }


class BaseFMZipModelForCausalLM(nn.Module, PushToHubMixin):
    layer_type: str = None
    layers_block_name: str = None
    outside_layer_modules: List[str] = None
    inside_layer_modules: List[List[str]] = None
    lm_head_name: str = "lm_head"

    fused_attn_module_type: Optional[FusedBaseAttentionModule] = None
    fused_mlp_module_type: Optional[FusedBaseMLPModule] = None

    def __init__(
        self,
        model: PreTrainedModel,
        compressed: bool,
        compress_config: BaseCompressionConfig,
    ):
        super().__init__()
        self.model = model
        self.model_type = self.model.config.model_type
        self._compressed = compressed
        self.compress_config = compress_config
        self.config = self.model.config

    @property
    def compressed(self):
        return self._compressed

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)

    @staticmethod
    def _resize_attention_mask(attention_mask: List[torch.LongTensor]):
        return attention_mask

    @staticmethod
    def _resize_position_ids(position_ids: List[torch.LongTensor]):
        return position_ids

    def _prepare_examples_for_compression(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
    ):
        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_examples = []
        for example in examples:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])
            if "labels" in example:
                labels = _convert_tensor_to_list(example["labels"])
            elif "label" in example:
                labels = _convert_tensor_to_list(example["label"])
            elif "label_ids" in example:
                labels = _convert_tensor_to_list(example["label_ids"])
            else:
                labels = copy.deepcopy(input_ids)
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        pad_token_id = self.config.pad_token_id
        if not pad_token_id:
            pad_token_id = self.config.eos_token_id
        new_examples = [
            collate_data(new_examples[start : start + batch_size], pad_token_id)
            for start in range(0, len(new_examples), batch_size)
        ]
        for new_example in new_examples:
            del new_example["labels"]

        return new_examples

    @torch.inference_mode()
    def lossy_compress(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        autotune_warmup_after_quantized: bool = False,
        cache_examples_on_gpu: bool = True,
    ):
        if self.compressed:
            raise EnvironmentError("Model is already compressed.")
        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    module = get_module_by_name(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    accelerate.cpu_offload_with_hook(module, CUDA_0)

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        examples = self._prepare_examples_for_compression(examples, batch_size)

        class LayerHijacker(nn.Module):
            """
            hijack layer's forward pass to cache data
            """

            def __init__(self, m, device):
                super().__init__()
                self.module = m
                self.data_device = device if cache_examples_on_gpu else CPU

            def forward(self, inp=None, **kwargs):
                if inp is None:
                    for kwarg_name in ["hidden_states"]:
                        if kwarg_name in kwargs:
                            inp = kwargs[kwarg_name]
                            break
                layer_inputs.append(move_to_device(inp, self.data_device))
                attention_masks.append(kwargs["attention_mask"].to(self.data_device))
                if (pos_ids := kwargs.get("position_ids", None)) is not None:
                    position_ids.append(move_to_device(pos_ids, self.data_device))
                one_kwargs = dict()
                for (
                    k,
                    v,
                ) in kwargs.items():  # make sure other arguments also be captured
                    if k not in ["hidden_states", "attention_mask", "position_ids"]:
                        if isinstance(v, torch.Tensor):
                            one_kwargs[k] = move_to_device(v, self.data_device)
                        else:
                            one_kwargs[k] = v
                layer_input_kwargs.append(one_kwargs)
                raise ValueError

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(examples)
        layers = get_module_by_name(self.model, self.layers_block_name)

        force_layer_back_to_cpu = False
        if get_device(layers[0]) == CPU:
            layers[0] = layers[0].to(CUDA_0)
            force_layer_back_to_cpu = True

        cur_layer_device = get_device(layers[0])
        ori_outside_layer_module_devices = {}
        for module_name in self.outside_layer_modules:
            module = get_module_by_name(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to_device(module, cur_layer_device)

        # get inputs for first layer
        layers[0] = LayerHijacker(layers[0], cur_layer_device)
        for example in examples:
            for k, v in example.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = move_to_device(v, cur_layer_device)
            try:
                self.model(**example)
            except ValueError:
                pass
        layers[0] = layers[0].module

        move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.outside_layer_modules:
            module = get_module_by_name(self.model, module_name)
            if module is not None:
                move_to_device(module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        # resize attention mask and position ids for some special models
        attention_masks = self._resize_attention_mask(attention_masks)
        position_ids = self._resize_position_ids(position_ids)

        inside_layer_modules = self.inside_layer_modules
        if not self.compress_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        compressors = {}
        for i in range(len(layers)):
            layer = layers[i]
            force_layer_back_to_cpu = False

            if get_device(layer) == CPU:
                move_to_device(layer, CUDA_0)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = find_layers(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names}
                sparsegpt = {}
                for name in subset:
                    sparsegpt[name] = SparseGPT(subset[name])
                    if self.compress_config.bits < 16:
                        sparsegpt[name].quantizer = Quantizer()
                        sparsegpt[name].quantizer.configure(
                            self.compress_config.bits,
                            perchannel=True,
                            sym=self.compress_config.sym,
                            mse=False,
                        )

                def add_batch(name):
                    def tmp(_, inp, out):
                        sparsegpt[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                    layer_attention_mask = move_to_device(
                        attention_masks[j], cur_layer_device
                    )
                    additional_layer_inputs = {"attention_mask": layer_attention_mask}
                    if (
                        layer_position_ids := None
                        if not position_ids
                        else move_to_device(position_ids[j], cur_layer_device)
                    ) is not None:
                        additional_layer_inputs["position_ids"] = layer_position_ids
                    for k, v in layer_input_kwargs[j].items():
                        if isinstance(v, torch.Tensor):
                            additional_layer_inputs[k] = move_to_device(
                                v, cur_layer_device
                            )
                        else:
                            additional_layer_inputs[k] = v
                    layer(layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                # starting compression
                for name in subset:
                    logger.debug(
                        f"Compression {name} in layer {i+1}/{len(layers)} - sparsity: {self.compress_config.sparsity}, bits: {self.compress_config.bits}"
                    )
                    scale, zero, g_idx = sparsegpt[name].fasterprune(
                        self.compress_config.sparsity,
                        prunen=self.compress_config.prunen,
                        prunem=self.compress_config.prunem,
                        percdamp=self.compress_config.damp_percent,
                        blocksize=self.compress_config.block_size,
                    )

                    compressors[f"{self.layers_block_name}.{i}.{name}"] = (
                        sparsegpt[name].quantizer.to(
                            CPU if force_layer_back_to_cpu else cur_layer_device
                        ),
                        move_to_device(
                            scale, CPU if force_layer_back_to_cpu else cur_layer_device
                        ),
                        move_to_device(
                            zero, CPU if force_layer_back_to_cpu else cur_layer_device
                        ),
                        move_to_device(
                            g_idx, CPU if force_layer_back_to_cpu else cur_layer_device
                        ),
                    )

                    sparsegpt[name].free()

            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                layer_attention_mask = move_to_device(
                    attention_masks[j], cur_layer_device
                )
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                if (
                    layer_position_ids := None
                    if not position_ids
                    else move_to_device(position_ids[j], cur_layer_device)
                ) is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                    else:
                        additional_layer_inputs[k] = v
                layer_output = move_to_device(
                    layer(layer_input, **additional_layer_inputs)[0],
                    cur_layer_device if cache_examples_on_gpu else CPU,
                )
                layer_outputs.append(layer_output)

            layers[i] = move_to_device(
                layer, CPU if force_layer_back_to_cpu else cur_layer_device
            )
            del layer
            del sparsegpt
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        self.compressors = compressors
        self.use_triton = use_triton
        self.use_cuda_fp16 = use_cuda_fp16
        self.autotune_warmup_after_quantized = autotune_warmup_after_quantized
        self.force_layer_back_to_cpu = force_layer_back_to_cpu

        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = accelerate.dispatch_model(
                self.model, device_map, offload_buffers=True
            )

        self.model.config.use_cache = forward_pass_use_cache
        self._compressed = True

        torch.cuda.empty_cache()

    @property
    def device(self):
        return self.model.device

    def to(self, device: Union[str, torch.device]):
        return self.model.to(device)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.inference_mode()
    def save_compressed(self, save_dir: str):
        if not self.compressed:
            raise EnvironmentError("Model is not compressed.")
        pack_model(
            model=self.model,
            quantizers=self.compressors,
            bits=self.compress_config.bits,
            group_size=self.compress_config.group_size,
            use_triton=self.use_triton,
            use_cuda_fp16=self.use_cuda_fp16,
            desc_act=self.compress_config.desc_act,
            warmup_triton=self.autotune_warmup_after_quantized,
            force_layer_back_to_cpu=self.force_layer_back_to_cpu,
        )
        os.makedirs(save_dir, exist_ok=True)
        self.model.to(CPU)
        model_save_name = f"fmzip-compressed.safetensors"
        state_dict = self.model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        if self.compress_config.lossless != "none":
            lossless_compressor = LosslessCompressor(self.compress_config.lossless)

            (
                state_dict,
                tensor_shapes,
                tensors_dtype,
            ) = lossless_compressor.compress_state_dict(state_dict)

        safe_save(
            tensor_dict=state_dict,
            filename=join(save_dir, model_save_name),
            metadata={"dtype": json.dumps(tensors_dtype), "shape": json.dumps(tensor_shapes)},
        )
        self.model.config.save_pretrained(save_dir)
        self.compress_config.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        compress_config: BaseCompressionConfig,
        max_memory: Optional[dict] = None,
        **model_init_kwargs,
    ):
        """load un-quantized pretrained model to cpu"""

        if not torch.cuda.is_available():
            raise EnvironmentError(
                "Load pretrained model to do quantization requires CUDA available."
            )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch.float16
        model_init_kwargs["trust_remote_code"] = True

        if max_memory:
            if "disk" in max_memory:
                raise NotImplementedError("disk offload not support yet.")
            with accelerate.init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.tie_weights()

            max_memory = accelerate.utils.get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
                low_zero=False,
            )
            model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
            )
            model_init_kwargs["low_cpu_mem_usage"] = True

            del model
        else:
            model_init_kwargs["device_map"] = None
            model_init_kwargs["low_cpu_mem_usage"] = False

        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_init_kwargs
        )
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning(
                "can't get model's sequence length from model config, will set to 4096."
            )
            model.seqlen = 4096
        model.eval()

        return cls(model, False, compress_config)

    @classmethod
    def from_compressed(
        cls,
        save_dir: str,
        device_map: Optional[str] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        strict: bool = False,
        use_triton: bool = False,
        inject_fused_attention: bool = False,
        inject_fused_mlp: bool = False,
        use_cuda_fp16: bool = True,
        compress_config: Optional[BaseCompressionConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = True,
        unpack: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ):
        """load compressed model from local disk"""
        config = AutoConfig.from_pretrained(
            save_dir, trust_remote_code=trust_remote_code
        )
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")
        if compress_config is None:
            compress_config = BaseCompressionConfig.from_pretrained(save_dir)

        if model_basename is None:
            model_basename = "fmzip-compressed"

        model_save_name = os.path.join(save_dir, model_basename)
        extensions = [".safetensors"]
        for ext in extensions:
            if isfile(model_save_name + ext):
                model_save_name += ext
                break
        else:
            raise FileNotFoundError(
                f"can't find model file with name {model_basename} in {save_dir}"
            )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False
        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights(include_buffers=False))

        with ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code, torch_dtype=torch.float)
            layers = find_layers(model)
            ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
            for name in list(layers.keys()):
                if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                    logger.info(
                        f"{name} not been quantized, will be ignored when make_quant."
                    )
                    del layers[name]
            make_quant(
                model,
                layers,
                compress_config.bits,
                compress_config.group_size,
                use_triton=use_triton,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=compress_config.desc_act,
            )
            model.tie_weights()
        if device is None and not device_map and not max_memory:
            device_map = "auto"
        if device is not None:
            device = torch.device(device)
            if not max_memory and not device_map:
                device_map = {
                    "": device.index if device.type == "cuda" else device.type
                }
        if not device_map:
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=[cls.layer_type]
            )
        # now load compressed data
        losslesscompressor = LosslessCompressor(compress_config.lossless)
        
        metadata = None
        tensors = {}
        
        with safe_open(model_save_name, framework='numpy') as f:
            metadata = f.metadata()
            keys = f.keys()
            for key in keys:
                tensors[key] = f.get_tensor(key)
        tensor_dtypes = json.loads(metadata["dtype"])
        tensor_shapes = json.loads(metadata["shape"])
        
        for key in tensors.keys():
            tensors[key] = cp.array(tensors[key], copy=False)
        tensors = losslesscompressor.decompress_state_dict(
            tensors, tensor_shapes, tensor_dtypes
        )
        model.load_state_dict(tensors, strict=False)
        if unpack:
            unpack_model(model)
        # set seqlen
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning(
                "can't get model's sequence length from model config, will set to 2048."
            )
            model.seqlen = 2048

        model.eval()
        return cls(model, True, compress_config)


__all__ = ["BaseFMZipModelForCausalLM", "BaseCompressionConfig"]
