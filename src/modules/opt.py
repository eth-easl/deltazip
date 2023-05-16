import copy
import torch
from loguru import logger
from src.dataloader import get_loaders
from transformers import OPTForCausalLM
from src.modules.common import auto_compress_generic, skip

quant_args = {
    'perchannel': True,
    'sym': True,
    'mse': False,
    'percdamp': 0.01,
    'maxshrink': 0.8,
    'trits': False,
    'grid': 100,
    'norm': 2.4,
    'groupsize': 1024,
    'actorder': False,
}

def get_opt_model(model_name: str, dtype=torch.float16):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.seqlen = model.config.max_position_embeddings
    return model

def get_opt_layers(model):
    return model.model.decoder.layers

def auto_compress(args):
    """
    The args will be the same as the ones in src/main.py
    """
    search_space = {
        'wbits': args.wbits,
        'sparsities': args.sparsities
    }
    logger.info("Loading models...")
    base_model = get_opt_model(args.base_model)
    target_model = get_opt_model(args.target_model)
    base_model.to('cuda')
    target_model.to('cuda')

    original_finetuned_model = copy.deepcopy(target_model)
    # now get delta model
    for base_p, finetuned_p in zip(base_model.parameters(), target_model.parameters()):
        finetuned_p.data = (finetuned_p.data - base_p.data).clone()
    logger.info("Loading data...")

    dataloader, testloader = get_loaders(
        args.dataset, seed=args.seed, model=args.target_model, seqlen=target_model.seqlen
    )
    logger.info("Starting compression...")
    
    auto_compress_generic(
        original_finetuned_model,
        target_model,
        dataloader,
        get_opt_layers,
        prep_model_func=None,
        tol=args.tol,
        search_space=search_space,
        n_samples=args.n_samples,
        quant_args=quant_args,
    )