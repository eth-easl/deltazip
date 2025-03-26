import torch
import torch.nn as nn
from vllm.delta.utils import (
    ext_gemm_half_q_half,
    ext_make_q_matrix,
    ExLlamaV2DeviceTensors,
)
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

    def post_init(self, temp_dq):
        if self.bits == 4:
            assert (
                self.qweight.device.type == "cuda"
            ), f"qweight must be on cuda devices, Found {self.qweight.device.type}"
            assert self.qweight.device.index is not None
            self.q_tensors = {
                "qweight": self.qweight,
                "qzeros": self.qzeros,
                "scales": self.scales,
                "g_idx": self.g_idx,
            }
            temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
            self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq=temp_dq)
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)

    @torch.inference_mode()
    def forward(self, x, y_slice):
        if self.bits == 4:
            output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, False)
            if self.bias:
                output.add_(self.bias)
            print(f"y_slice.shape: {y_slice.shape}, output.shape: {output.shape}")
            print(output)
            y_slice += output
            return output
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    @classmethod
    def from_tensors(cls, qweight, qzeros, scales, g_idx, bias, device_tensor):
        # TODO(xiaozhe): debug only, fix later
        bits = 4
        infeatures = qweight.shape[0] * 32 // bits
        outfeatures = qweight.shape[1]
        obj = cls(bits, infeatures, outfeatures, bias)
        obj.qweight = qweight
        obj.qzeros = qzeros
        obj.scales = scales
        obj.g_idx = g_idx
        # TODO(xiaozhe): here we need the post_init function
        obj.post_init(temp_dq=device_tensor)
        return obj
