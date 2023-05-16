import torch
import torch.nn as nn
from loguru import logger
from src.gptq import GPTQ
from typing import Callable
from src.quant import Quantizer
from src.modules.utils import find_layers

def skip(*args, **kwargs):
    pass

BASE_FLOATS = 16

@torch.no_grad()
def auto_compress_generic(
        model,
        delta_model,
        dataloader,
        get_layers_func: Callable,
        prep_model_func: Callable,
        tol=0.2,
        search_space=None,
        n_samples=128,
        quant_args = {},
    ):
    compression_rates = {}
    masks = {}
    if delta_model is None:
        raise NotImplementedError
    # populate compression rate with different search options
    for wbit in search_space['wbits']:
        for sparsity in search_space['sparsities']:
            compression_rates[f'wbit={wbit}_sparsity={sparsity}'] = (BASE_FLOATS / wbit) / (1 - sparsity)
    compression_rates = sorted(
        compression_rates.items(),
        key=lambda x: x[1],
        reverse=True
    )
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers_func(model)
    delta_layers = get_layers_func(delta_model)
    if prep_model_func:
        prep_model_func(model)
    dev = model.device
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (n_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device=model.device
    )
    cache = {"i": 0, "attention_mask": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    # restore the original layer
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    original_outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    tune_obj = {}
    tuned_params = {}
    quantizers = {}
    for i in range(len(layers)):
        layer = delta_layers[i].to(dev)
        original_layer = layers[i].to(dev)
        subset = find_layers(layer)
        
        for name in subset:
            # configuring quantizers
            tuned_params[f"{i}_{name}"] = {}
            tune_obj[f"{i}_{name}"] = {}
            for wbit in search_space['wbits']:
                tune_obj[f"{i}_{name}"][f'wbit={wbit}'] = {
                    'gptq': GPTQ(subset[name]),
                }
                tune_obj[f"{i}_{name}"][f'wbit={wbit}']['gptq'].quantizer = Quantizer()
                tune_obj[f"{i}_{name}"][f'wbit={wbit}']['gptq'].quantizer.configure(
                    bits=wbit,
                    perchannel=quant_args['perchannel'],
                    mse=quant_args['mse'],
                    norm=quant_args['norm'],
                    grid=quant_args['grid'],
                    maxshrink=quant_args['maxshrink'],
                    trits=quant_args['trits']
                )
        
        def add_batch(name):
            def tmp(_, inp, out):
                for wbit in search_space['wbits']:
                    tune_obj[f'{i}_{name}'][f'wbit={wbit}']['gptq'].add_batch(inp[0].data, out.data)
            return tmp
        
        # adding input hooks
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(n_samples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            original_outs[j] = original_layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask
            )[0]

        for h in handles:
            h.remove()
        
        for name in subset:
            logger.info(f"Quantizing {i}.{name} ...")
            for wbit in search_space['wbits']:
                logger.debug("Searching for Optimal Quantization Parameters")
                losses, _ = tune_obj[f'{i}_{name}'][f'wbit={wbit}']['gptq'].fasterquant(
                    percdamp=quant_args['percdamp'],
                    groupsize=quant_args['groupsize'],
                    actorder=quant_args['actorder'],
                    sparsity = search_space['sparsities'],
                    write=False,
                )
                for s_sity in losses.keys():
                    tuned_params[f'{i}_{name}'][f'wbit={wbit}_sparsity={s_sity}'] = {
                        'loss': losses[s_sity].item()
                    }
                    logger.debug(f"wbit: {wbit}; sparsity: {s_sity}; loss: {losses[s_sity].item()}")
            best_wbit = None
            best_sparsity = None
            best_loss = None
            # starting from the maximal compression rate
            # and going down until the loss is within tolerance
            for cr in compression_rates:
                config = cr[0]
                wbit = int(config.split('_')[0].split('=')[1])
                sparsity = float(config.split('_')[1].split('=')[1])
                # find the corresponding loss
                loss = tuned_params[f'{i}_{name}'][f'wbit={wbit}_sparsity={sparsity}']['loss']
                if loss <= tol:
                    best_wbit = wbit
                    best_sparsity = sparsity
                    best_loss = loss
                    break
            # if it's still None, then we didn't find a good enough compression rate, we fall back to the minimal compression rate
            if best_wbit is None:
                best_wbit = compression_rates[-1][0].split('_')[0].split('=')[1]
                best_sparsity = compression_rates[-1][0].split('_')[1].split('=')[1]
            best_loss = tuned_params[f'{i}_{name}'][f'wbit={best_wbit}_sparsity={best_sparsity}']['loss']
            # now we find the optimal parameters, apply it to the model
            logger.debug(f"Found optimal quantization parameters for {i}.{name}: wbit={best_wbit}, sparsity={best_sparsity}, loss={best_loss}")
            loss, mask = tune_obj[f'{i}_{name}'][f'wbit={best_wbit}']['gptq'].fasterquant(
                percdamp=quant_args['percdamp'],
                groupsize=quant_args['groupsize'],
                actorder=quant_args['actorder'],
                write=True,
                sparsity = [best_sparsity],
            )
            if mask is not None:
                masks[f'{i}_{name}'] = mask

            quantizers[f"{i}.{name}"] = tune_obj[f'{i}_{name}'][f'wbit={best_wbit}']['gptq'].quantizer
            tune_obj[f"{i}_{name}"][f"wbit={best_wbit}"]['gptq'].free()
            tuned_params[f"{i}_{name}"]['choice'] = {
                'best_wbit': best_wbit,
                'best_sparsity': best_sparsity,
                'best_loss': best_loss,
            }
        for j in range(n_samples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            original_outs[j] = original_layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask
            )[0]
        del layer
        for key in tuned_params.keys():
            if key.startswith(f'{i}_'):
                for wbit in search_space['wbits']:
                    del tune_obj[key][f'wbit={wbit}']['gptq']
        torch.cuda.empty_cache()
        inps, outs = original_outs, inps

    model.config.use_cache = use_cache
    return quantizers, tuned_params, masks