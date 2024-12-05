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
from safetensors.torch import save_file as safe_torch_save
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
from ..core.quant import Quantizer
from ..core.sparsegpt import SparseGPT
from ..utils.data_utils import collate_data
from ..nn_modules.qlinear_cuda import QuantLinear
from deltazip.modeling._utils import deltazip_post_init
from deltazip.utils.converter import convert_model
try:
    from ..lossless.compressor import LosslessCompressor
except ImportError:
    LosslessCompressor = None
    print("LosslessCompressor not found")

triton_has_warmup = False
NUM_DEBUG_LAYER = 5


@dataclass
class AutoCompressionConfig(PushToHubMixin):
    tolerance: float = field(default=1e-9)
    bits: List[int] = field(default_factory=lambda: [2, 3, 4, 8])
    sparsity: List[float] = field(default_factory=[0, 0.5, 0.75, 0.9, 0.99])
    prunen: int = field(default=0)
    prunem: int = field(default=0)
    block_size: int = field(default=128)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    lossless: str = field(default="none")
    dtype: str = field(default="fp16")
    final_bit: Dict[str, int] = field(default_factory=lambda: {})
    final_sparsity: Dict[str, float] = field(default_factory=lambda: {})

    def __post_init__(self):
        for bit in self.bits:
            if bit not in [2, 3, 4, 8]:
                raise ValueError(f"bit must be one of [2,3,4,8]. Got {bit}")
        for sparsity in self.sparsity:
            if not (0 <= sparsity <= 1):
                raise ValueError(
                    f"sparsity must between 0 and 1. Got {sparsity}")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(
            join(save_dir, "auto_compress_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str):
        with open(
            join(save_dir, "auto_compress_config.json"), "r", encoding="utf-8"
        ) as f:
            return cls(**json.load(f))

    def to_dict(self):
        return {
            "bits": self.bits,
            "sparsity": self.sparsity,
            "final_bit": self.final_bit,
            "final_sparsity": self.final_sparsity,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "lossless": self.lossless,
            "prunen": self.prunen,
            "prunem": self.prunem,
            "block_size": self.block_size,
        }


@dataclass
class BaseCompressionConfig(PushToHubMixin):
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8, 16]})
    # sparsity = how many parameters we set to zero after quantization
    sparsity: float = field(default=0)
    prunen: int = field(default=0)
    prunem: int = field(default=0)
    group_size: int = field(default=-1)
    # deprecated, for backward compatibility
    group_rows: int = field(default=-1)
    block_size: int = field(default=128)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=False)
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
            raise ValueError(
                "unless equal to -1, group_size must greater then 0.")
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
            "sparsity": self.sparsity,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "lossless": self.lossless,
            "prunen": self.prunen,
            "prunem": self.prunem,
            "block_size": self.block_size,
        }


class BaseDeltaZipModelForCausalLM(nn.Module, PushToHubMixin):
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
        compress_config: Union[AutoCompressionConfig, BaseCompressionConfig],
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
            collate_data(new_examples[start: start + batch_size], pad_token_id)
            for start in range(0, len(new_examples), batch_size)
        ]
        for new_example in new_examples:
            del new_example["labels"]

        return new_examples

    @torch.inference_mode()
    def lossless_compress(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        autotune_warmup_after_quantized: bool = False,
        cache_examples_on_gpu: bool = False,
    ):
        self._compressed = True

    @torch.inference_mode()
    def lossy_compress(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        autotune_warmup_after_quantized: bool = False,
        cache_examples_on_gpu: bool = False,
        base_model=None,
    ):
        assert self.compressed == False, "Model is already compressed."
        logger.info(f"Compression Config: {self.compress_config}")
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
                
                if kwargs["attention_mask"] is not None:
                    attention_masks.append(kwargs["attention_mask"].to(self.data_device))
                else:
                    attention_masks.append(None)
                
                if (pos_ids := kwargs.get("position_ids", None)) is not None:
                    position_ids.append(move_to_device(
                        pos_ids, self.data_device))
                one_kwargs = dict()
                for (
                    k,
                    v,
                ) in kwargs.items():
                    # make sure other arguments also be captured
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
            except ValueError as e:
                pass

        layers[0] = layers[0].module
        move_to_device(
            layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.outside_layer_modules:
            module = get_module_by_name(self.model, module_name)
            if module is not None:
                move_to_device(
                    module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        # resize attention mask and position ids for some special models
        attention_masks = self._resize_attention_mask(attention_masks)
        position_ids = self._resize_position_ids(position_ids)

        inside_layer_modules = self.inside_layer_modules
        if not self.compress_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        self.compressors = {}
        compressed_ws = {}
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
                    handles.append(
                        subset[name].register_forward_hook(add_batch(name)))

                for j in range(num_batches):
                    layer_input = move_to_device(
                        layer_inputs[j], cur_layer_device)
                    
                    layer_attention_mask = move_to_device(
                        attention_masks[j], cur_layer_device
                    )
                    additional_layer_inputs = {
                        "attention_mask": layer_attention_mask}
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
                    if base_model is not None:
                        base_weight = base_model.model.state_dict()[
                            f"{self.layers_block_name}.{i}.{name}.weight"
                        ]
                        base_weight = move_to_device(
                            base_weight, cur_layer_device)
                    scale, zero, g_idx, avg_loss, compressed_w = sparsegpt[
                        name
                    ].fasterprune(
                        sparsity=self.compress_config.sparsity,
                        prunen=self.compress_config.prunen,
                        prunem=self.compress_config.prunem,
                        percdamp=self.compress_config.damp_percent,
                        blocksize=self.compress_config.block_size,
                        actorder=self.compress_config.desc_act,
                        base_weight=base_weight if base_model is not None else None,
                    )
                    if self.compress_config.bits < 16:
                        self.compressors[f"{self.layers_block_name}.{i}.{name}"] = (
                            sparsegpt[name].quantizer.to(
                                CPU if force_layer_back_to_cpu else cur_layer_device
                            ),
                            move_to_device(
                                scale,
                                CPU if force_layer_back_to_cpu else cur_layer_device,
                            ),
                            move_to_device(
                                zero,
                                CPU if force_layer_back_to_cpu else cur_layer_device,
                            ),
                            move_to_device(
                                g_idx,
                                CPU if force_layer_back_to_cpu else cur_layer_device,
                            ),
                        )
                        # move it back to cpu to save memory
                        assert (
                            f"{self.layers_block_name}.{i}.{name}" not in compressed_ws
                        )

                        compressed_ws[
                            f"{self.layers_block_name}.{i}.{name}"
                        ] = compressed_w.to(CPU).clone()

                        sparsegpt[name].free()
                        if base_model is not None:
                            del base_weight

            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                layer_attention_mask = move_to_device(
                    attention_masks[j], cur_layer_device
                )
                additional_layer_inputs = {
                    "attention_mask": layer_attention_mask}
                if (
                    layer_position_ids := None
                    if not position_ids
                    else move_to_device(position_ids[j], cur_layer_device)
                ) is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(
                            v, cur_layer_device)
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

        self.use_triton = use_triton
        self.use_cuda_fp16 = use_cuda_fp16
        self.autotune_warmup_after_quantized = autotune_warmup_after_quantized
        self.force_layer_back_to_cpu = force_layer_back_to_cpu

        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = accelerate.dispatch_model(
                self.model, device_map, offload_buffers=True
            )
        logger.info("Compress finished... moving compressed delta back")
        if base_model is not None:
            for i in range(len(layers)):
                # move compressed weights back
                full = find_layers(layers[i])
                for names in inside_layer_modules:
                    subset = {n: full[n] for n in names}
                    for name in subset:
                        if self.compress_config.bits < 16:
                            finetuned_weight = subset[name].weight.data
                            delta_only = compressed_ws[
                                f"{self.layers_block_name}.{i}.{name}"
                            ]
                            # base_weight = base_model.model.state_dict()[
                            #     f"{self.layers_block_name}.{i}.{name}.weight"
                            # ]
                            # assert torch.equal(
                            #     finetuned_weight, base_weight + delta_only
                            # )
                            key_weight = compressed_ws[
                                f"{self.layers_block_name}.{i}.{name}"
                            ]
                            if subset[name].weight.is_meta:
                                subset[name].weight = torch.nn.Parameter(
                                    key_weight.clone().detach(), requires_grad=False).to(CPU)
                            else:
                                subset[name].weight.copy_(compressed_ws[
                                    f"{self.layers_block_name}.{i}.{name}"
                                ])

        for name, param in self.model.named_parameters():
            print(f"{name}: {param.device}")
        self.model.config.use_cache = forward_pass_use_cache
        self._compressed = True
        torch.cuda.empty_cache()

    @property
    def device(self):
        return self.model.device

    def to(self, device: Union[str, torch.device], non_blocking: bool = False):
        return self.model.to(device, non_blocking=non_blocking)

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
        if isinstance(
            self.compress_config, AutoCompressionConfig
        ) or self.compress_config.bits in [2, 3, 4, 8]:
            pack_model(
                model=self.model,
                quantizers=self.compressors,
                bits=self.compress_config.bits
                if isinstance(self.compress_config.bits, int)
                else self.compress_config.final_bit,
                use_triton=self.use_triton,
                use_cuda_fp16=self.use_cuda_fp16,
                desc_act=self.compress_config.desc_act,
                warmup_triton=self.autotune_warmup_after_quantized,
                force_layer_back_to_cpu=self.force_layer_back_to_cpu,
            )
        os.makedirs(save_dir, exist_ok=True)
        self.model.to(CPU)
        model_save_name = f"deltazip-compressed.safetensors"
        state_dict = self.model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        if self.compress_config.lossless != "none":
            lossless_compressor = LosslessCompressor(
                self.compress_config.lossless)
            (
                state_dict,
                tensor_shapes,
                tensors_dtype,
            ) = lossless_compressor.compress_state_dict(state_dict)
            safe_save(
                tensor_dict=state_dict,
                filename=join(save_dir, model_save_name),
                metadata={
                    "dtype": json.dumps(tensors_dtype),
                    "shape": json.dumps(tensor_shapes),
                },
            )
        else:
            if self.compress_config.prunen == 2 and self.compress_config.prunem == 4 and self.compress_config.bits==4:
                logger.info("bits=4, prune n=2, m=4, will save model with structured sparse format")
                state_dict = convert_model(state_dict, verbose=True)
            # we use safe_torch_save to save model, since it is not losslessly compressed
            safe_torch_save(
                tensors=state_dict,
                filename=join(save_dir, model_save_name),
            )
        
        self.model.config.save_pretrained(save_dir)
        self.compress_config.save_pretrained(save_dir)

    @classmethod
    def from_lora(
        cls,
        pretrained_model_name_or_path: str,
        compress_config: BaseCompressionConfig,
        max_memory: Optional[dict] = None,
        **model_init_kwargs,
    ):
        """load lora fine-tuned model"""
        if not torch.cuda.is_available():
            raise EnvironmentError("Load LoRA model requires CUDA available.")

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
        model_init_kwargs["torch_dtype"] = torch.float16
        model_init_kwargs["trust_remote_code"] = True
        torch.cuda.empty_cache()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        compress_config: BaseCompressionConfig,
        max_memory: Optional[dict] = None,
        device_map: Optional[str] = None,
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
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True)
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
            if device_map is None:
                model_init_kwargs["device_map"] = None
            else:
                model_init_kwargs["device_map"] = device_map

            logger.info(
                f"Using [{model_init_kwargs['device_map']}] to load model.")
            # model_init_kwargs["low_cpu_mem_usage"] = True

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
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        compress_config: Optional[
            Union[BaseCompressionConfig, AutoCompressionConfig]
        ] = None,
        model_basename: Optional[str] = None,
        trust_remote_code: bool = False,
        unpack: bool = False,
        low_cpu_mem_usage: bool = False,
        use_bfloat16: bool = False,
        use_exllama: bool = False,
        **kwargs,
    ):
        """load compressed model from local disk"""
        config = AutoConfig.from_pretrained(
            save_dir, trust_remote_code=trust_remote_code
        )
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")
        if compress_config is None:
            # check if "auto_compression_config.json" exists
            if isfile(join(save_dir, "auto_compress_config.json")):
                compress_config = AutoCompressionConfig.from_pretrained(
                    save_dir)
            else:
                compress_config = BaseCompressionConfig.from_pretrained(
                    save_dir)
        logger.info(f"compress config: {compress_config}")

        if model_basename is None:
            model_basename = "deltazip-compressed"

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
        # if low_cpu_mem_usage:
        #     init_contexts.append(accelerate.init_empty_weights(include_buffers=False))
        if compress_config.lossless != "none":
            if isinstance(
                compress_config, AutoCompressionConfig
            ) or compress_config.bits in [2, 3, 4, 8]:
                with ContextManagers(init_contexts):
                    model = AutoModelForCausalLM.from_config(
                        config,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch.float16,
                    )
                    layers = find_layers(model)
                    ignore_layers = [cls.lm_head_name] + \
                        cls.outside_layer_modules
                    for name in list(layers.keys()):
                        if any(
                            [
                                name.startswith(ignore_layer)
                                for ignore_layer in ignore_layers
                            ]
                        ):
                            logger.info(
                                f"{name} not been quantized, will be ignored when make_quant."
                            )
                            del layers[name]
                    make_quant(
                        model,
                        layers,
                        bits=compress_config.bits
                        if isinstance(compress_config, BaseCompressionConfig)
                        else compress_config.final_bit,
                        use_triton=use_triton,
                        use_cuda_fp16=use_cuda_fp16,
                        desc_act=compress_config.desc_act,
                        use_exllama=use_exllama,
                    )
                    model.tie_weights()
                if device is None and not device_map and not max_memory:
                    device_map = "auto"
                if device is not None:
                    device = torch.device(device)
            else:
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.float16,
                )
            # now load compressed data
            losslesscompressor = LosslessCompressor(
                compress_config.lossless, device_id=0)
            metadata = None
            tensors = {}

            with safe_open(model_save_name, framework="numpy") as f:
                metadata = f.metadata()
                keys = f.keys()
                for key in keys:
                    tensors[key] = f.get_tensor(key)
            tensor_dtypes = json.loads(metadata["dtype"])
            tensor_shapes = json.loads(metadata["shape"])
            # (todo: xiaozhe), (todo: minor)
            # seems like we cannot use arbitrary device to decompress
            # for now use device=0 to decompress and then move to target device
            with cp.cuda.Device(0):
                for key in tensors.keys():
                    tensors[key] = cp.array(tensors[key], copy=False)
            tensors = losslesscompressor.decompress_state_dict(
                tensors,
                tensor_shapes,
                tensor_dtypes,
                use_bfloat16=use_bfloat16,
                target_device=device,
            )
        else:
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
            )
            layers = find_layers(model)
            ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
            for name in list(layers.keys()):
                if any(
                    [
                        name.startswith(ignore_layer)
                        for ignore_layer in ignore_layers
                    ]
                ):
                    logger.info(
                        f"{name} not been quantized, will be ignored when make_quant."
                    )
                    del layers[name]
            make_quant(
                model,
                layers,
                bits=compress_config.bits
                if isinstance(compress_config, BaseCompressionConfig)
                else compress_config.final_bit,
                use_triton=use_triton,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=compress_config.desc_act,
                use_exllama=use_exllama,
            )
            tensors = {}
            with safe_open(model_save_name, framework="pt") as f:
                metadata = f.metadata()
                keys = f.keys()
                for key in keys:
                    tensors[key] = f.get_tensor(key)
        # move tensors to target device
        # print model keys
        missing_keys, unexpected_keys = model.load_state_dict(
            tensors, strict=False, assign=True
        )
        if missing_keys:
            logger.debug(f"missing keys: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"unexpected keys: {unexpected_keys}")
        model = model.to(device)
        model = deltazip_post_init(
            model, use_act_order=compress_config.desc_act)
        model.eval()
        if compress_config.lossless != "none":
            if isinstance(
                compress_config, AutoCompressionConfig
            ) or compress_config.bits in [2, 3, 4, 8]:
                del tensor_dtypes
                del tensor_shapes
                del tensors
                del layers
            if unpack and (
                isinstance(compress_config, AutoCompressionConfig)
                or compress_config.bits in [2, 3, 4, 8]
            ):
                unpack_model(model)
            del losslesscompressor
        else:
            if unpack and (
                isinstance(compress_config, AutoCompressionConfig)
                or compress_config.bits in [2, 3, 4, 8]
            ):
                unpack_model(model)
            # print keys in the model
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

        global triton_has_warmup
        if not triton_has_warmup and use_triton:
            QuantLinear.warmup(model, seqlen=model.seqlen)
            triton_has_warmup = True
        torch.cuda.empty_cache()
        return cls(model, True, compress_config)


__all__ = ["BaseDeltaZipModelForCausalLM", "BaseCompressionConfig"]