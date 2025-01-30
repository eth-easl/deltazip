import torch
from tqdm import tqdm
from triteia.python.ops.utils.generator import torch_weight_to_sparse_marlin
from triteia.python.utils.io import save_tensors
from triteia.python.utils.quant_utils import dequantize_weight
from triteia.python.configs.models.llama import (
    row_chunking_modules,
    uncompressed_row_chunking_modules,
    pack_modules,
)

@torch.no_grad()
def convert_model(tensors, verbose=True):
    DEV = "cuda:0"
    tp_size = 1
    new_tensors = {}
    packed_tensors = {}
    dequantized_tensors = {}
    remaining_keys = list(tensors.keys())
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    pbar = tqdm(quantized_modules, position=0, leave=True)
    print("Dequantizing weights...")
    for module in pbar:
        dequantized_weight = (
            dequantize_weight(
                tensors[module + ".qweight"],
                tensors[module + ".qzeros"],
                tensors[module + ".scales"],
            )
            .to(torch.float16)
            .t()
            .cpu()
        )
        scales = tensors[module + ".scales"]
        dequantized_tensors[module] = (dequantized_weight, scales)
        remaining_keys.remove(module + ".qweight")
        remaining_keys.remove(module + ".qzeros")
        remaining_keys.remove(module + ".scales")
        remaining_keys.remove(module + ".g_idx")
    pack_plan = {}
    for module in quantized_modules:
        if any([key in module for key in pack_modules.keys()]):
            source_layer = module.rsplit(".", 2)[0]
            source_module = module.replace(source_layer + ".", "")
            target_module = pack_modules[source_module]
            target_idx = int(target_module.split(":")[1])
            target_module = source_layer + "." + target_module.split(":")[0]
            if target_module not in pack_plan:
                pack_plan[target_module] = []
            pack_plan[target_module].append((module, target_idx))

        elif any([key in module for key in row_chunking_modules]):
            qweights, scales, metas = torch_weight_to_sparse_marlin(
                dequantized_tensors[module][0].to(DEV),
                dequantized_tensors[module][1].to(DEV),
                tp_size=tp_size,
                chunk_by="row",
            )
            for idx, (qweight, scales, meta) in enumerate(zip(qweights, scales, metas)):
                new_tensors[module + f".{idx}.qweight"] = qweight
                new_tensors[module + f".{idx}.scales"] = scales
                new_tensors[module + f".{idx}.meta"] = meta
    for key in pack_plan.keys():
        key_weights = []
        key_scales = []
        plan = sorted(pack_plan[key], key=lambda x: x[1])
        for module, idx in plan:
            weight, scales = dequantized_tensors[module]
            assert weight.shape[1] == scales.shape[1]
            key_weights.append(weight)
            key_scales.append(scales)
        key_weights = torch.cat(key_weights, dim=1)
        key_scales = torch.cat(key_scales, dim=1)
        packed_tensors[key] = (key_weights, key_scales)
        torch.cuda.synchronize()
        del dequantized_tensors[module]
        torch.cuda.empty_cache()

        qweights, scales, metas = torch_weight_to_sparse_marlin(
            packed_tensors[key][0].to(DEV),
            packed_tensors[key][1].to(DEV),
            tp_size=tp_size,
            chunk_by="column",
        )
        for idx, (qweight, scales, meta) in enumerate(zip(qweights, scales, metas)):
            new_tensors[key + f".{idx}.qweight"] = qweight
            new_tensors[key + f".{idx}.scales"] = scales
            new_tensors[key + f".{idx}.meta"] = meta
    for module in remaining_keys:
        if any([key in module for key in uncompressed_row_chunking_modules]):
            weight = tensors[module]
            module_name = module.removesuffix(".weight")
            num_rows = weight.shape[0]
            for i in range(tp_size):
                tp_weight = weight[
                    i * num_rows // tp_size : (i + 1) * num_rows // tp_size, :
                ]
                new_tensors[module_name + f".{i}.weight"] = tp_weight
    return new_tensors
