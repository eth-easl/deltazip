import math
import time

import torch
import torch.nn as nn
import transformers
from loguru import logger

from .quant import quantize, Quantizer

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def calculate_sparsity(tensor: torch.Tensor):
    return (tensor == 0).sum().item() / tensor.numel()


class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=0.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        # we repeat the process and find percdamp that doesn't cause instability
        success_damp = False
        while not success_damp:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
            # check if Hinv contains nan
            if not torch.isnan(Hinv).any():
                success_damp = True
            else:
                logger.warning(f"NaN in Hinv, increasing percdamp to {percdamp + 0.01}")
                percdamp = percdamp + 0.01
                if percdamp >= 0.05:
                    raise ValueError("percdamp too high (>=0.05), aborting")

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
                    # sparsity: mask1 is a boolean mask, True means the weight is pruned
                    # the larger the sparsity, the more weights are pruned (higher compression ratio)
                    if sparsity == 0:
                        thresh = -9999
                    else:
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp < thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = w.clone()
                # q[mask1[:, i]] = 0
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

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = W[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        avg_loss = torch.sum(Losses).item() / self.nsamples
        logger.info(f"duration: {(time.time() - tick)}")
        logger.info(f"avg loss: {avg_loss}")
        logger.info(f"sparsity: {calculate_sparsity(W)}")

        g_idx = [i // self.columns for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=W.device)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        if scale == [] and hasattr(self, "quantizer"):
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        if hasattr(self, "quantizer"):
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, avg_loss

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
