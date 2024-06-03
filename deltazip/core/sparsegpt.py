import math
import time

import torch
import torch.nn as nn
import transformers
from loguru import logger

from .quant import quantize
from .sparsity_utils import calculate_sparsity

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        self.inp1 = inp
        self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        sparsity_H = calculate_sparsity(self.H)
        if sparsity_H == 1:
            raise ValueError("sparsity of H == 1, something is off, aborting")

    def fasterprune(
        self,
        sparsity,
        prunen=0,
        prunem=0,
        blocksize=128,
        percdamp=0.01,
        actorder=False,
        base_weight=None,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        if base_weight is not None:
            base_weight = base_weight.float()
            logger.info(f"compression operates on delta...")

            assert (
                base_weight.shape == W.shape
            ), "base_weight shape should be the same as W"
            W -= base_weight
        before_sparsity = calculate_sparsity(W)
        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        Losses = torch.zeros(self.rows, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # check if Hinv contains nan
        if torch.isnan(Hinv).any():
            raise ValueError("Hinv contains nan, aborting...")
        g_idx = []
        scale = []
        zero = []
        mask = None
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    # sparsity: mask1 is a boolean mask, True == weight pruned
                    # the larger the sparsity, the more weights are pruned (=> higher compression ratio)
                    if sparsity == 0:
                        thresh = -9999
                    else:
                        thresh = torch.sort(tmp.flatten())[0][
                            int(tmp.numel() * sparsity)
                        ]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
                    
                q = w.clone()
                q[mask1[:, i]] = 0
                if hasattr(self, "quantizer"):
                    q = quantize(
                        q.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1

            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            if DEBUG:
                if base_weight is not None:
                    self.layer.weight.data[:, :i2] = W[:, :i2] + base_weight[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:] + base_weight[:, i2:]
                else:
                    self.layer.weight.data[:, :i2] = W[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(f"block {i1} - {i2}")
                logger.debug(
                    f"reconstruct loss: {torch.sum((self.layer(self.inp1) - self.out1) ** 2)}"
                )
                logger.debug(f"loss: {torch.sum(Losses1)}")

        torch.cuda.synchronize()
        g_idx = [i // self.columns for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=W.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
        W = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if base_weight is not None:
            logger.debug("adding base weight for correct forward...")
            # set the layer's weight to be (compressed) W + (uncompressed) base_weight
            # such that the next layer has a signal of compressed delta
            self.layer.weight.data = (W + base_weight).to(self.layer.weight.data.dtype)
        else:
            # if base_weight is None
            # --> we compress the whole model
            # --> set the layer's weight to be compressed W
            self.layer.weight.data = W.to(self.layer.weight.data.dtype)
        after_sparsity = calculate_sparsity(W)
        logger.info(f"duration: {(time.time() - tick):.2f}s")
        logger.info(f"sparsity: {after_sparsity}")

        avg_loss = torch.sum(Losses).item() / self.nsamples
        if avg_loss >= 0.5:
            logger.warning(f"High avg loss detected: {avg_loss}")
        else:
            logger.info(f"avg loss: {avg_loss}")
        # reconstruct_loss = torch.mean((self.layer(self.inp1) - self.out1) ** 2)
        # logger.info(f"rec loss: {reconstruct_loss}")
        if after_sparsity - before_sparsity > 0.5:
            logger.warning(
                f"high sparsity change detected: {before_sparsity} -> {after_sparsity}"
            )

        if scale == [] and hasattr(self, "quantizer"):
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        if hasattr(self, "quantizer"):
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, avg_loss, W

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
