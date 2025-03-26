import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class QuantLinear(nn.Module):

    def __init__(
        self,
        bits: int,
        infeatures: int,
        outfeatures: int,
        bias,
    ):
        super().__init__()
        global _autogptq_cuda_available
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = infeatures
        self.maxq = 2**self.bits - 1
        self.qweight = None
        self.qzeros = None
        self.scales = None
        self.g_idx = None
        self.bias = bias
        if self.bits == 4:
            self.padding = -outfeatures % 32
        self.wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    @torch.inference_mode()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                self.wf.unsqueeze(0),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2**self.bits) - 1, out=zeros)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(weight, (2**self.bits) - 1, out=weight)

        elif self.bits == 3:
            raise NotImplementedError("3 bits are not supported.")

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        assert num_itr == 1, "num_itr must be 1"
        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (
                weight - zeros[self.g_idx.long()]
            )
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                weights.append(
                    scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()])
                )
            weights = torch.cat(weights, dim=1)
        out = torch.matmul(x.half(), weights)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    @classmethod
    def from_tensors(
        cls, bitwidth, qweight, qzeros, scales, g_idx, bias, device_tensor
    ):
        infeatures = qweight.shape[0] * 32 // bitwidth
        outfeatures = qweight.shape[1]
        obj = cls(bitwidth, infeatures, outfeatures, bias)
        obj.qweight = qweight
        obj.qzeros = qzeros.to(qweight.device)
        obj.scales = scales.to(qweight.device)
        obj.g_idx = g_idx.to(qweight.device)
        return obj
