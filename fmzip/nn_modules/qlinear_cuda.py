import math
import torch
import numpy as np
import torch.nn as nn
import transformers
from loguru import logger

from fmzip.nn_modules.triton_utils.kernels import (
    quant_matmul_inference_only_248,
    QuantLinearInferenceOnlyFunction,
)

try:
    import autogptq_cuda_256
    import autogptq_cuda_64

    _autogptq_cuda_available = True
except ImportError:
    logger.warning("CUDA extension not installed.")
    autogptq_cuda_256 = None
    autogptq_cuda_64 = None
    _autogptq_cuda_available = False

try:
    from fmzip.nn_modules.exllama_utils import ext_make_q_matrix, ext_gemm_half_q_half

    _exllama_v2_available = True
except ImportError as exllama_v2_import_exception:
    logger.warning("Exllama V2 extension not installed.")
    _exllama_v2_available = False

    def error_raiser_exllama(*args, **kwargs):
        raise ValueError(
            f"Trying to use the exllama v2 backend, but could not import the C++/CUDA dependencies with the following error: {exllama_v2_import_exception}"
        )

    make_q_matrix = error_raiser_exllama
    gemm_half_q_half = error_raiser_exllama

# dummy tensor
none_tensor = torch.empty((1, 1), device="meta")


class QuantLinear(nn.Module):
    QUANT_TYPE = "cuda"

    def __init__(
        self,
        bits,
        infeatures,
        outfeatures,
        bias,
        kernel_switch_threshold=256,
        trainable=False,
        use_triton=True,
        use_exllama=False,
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
        self.use_exllama = use_exllama
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros", torch.zeros((1, outfeatures // 32 * self.bits), dtype=torch.int32)
        )
        self.register_buffer(
            "scales", torch.zeros((1, outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [i // infeatures for i in range(infeatures)], dtype=torch.int32
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=torch.float16)
            )
        else:
            self.bias = None

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(
                list(range(0, 32, self.bits)), dtype=torch.int32
            ).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.autogptq_cuda_available = _autogptq_cuda_available

        self.autogptq_cuda = autogptq_cuda_256
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            self.autogptq_cuda = autogptq_cuda_64
        if infeatures % 64 != 0 or outfeatures % 64 != 0:
            self.autogptq_cuda_available = False
        self.use_triton = use_triton
        self.trainable = trainable
        if self.bits == 4 and _exllama_v2_available and self.use_exllama:
            self.padding = -outfeatures % 32

    def post_init(self, temp_dq):
        if self.bits == 4 and _exllama_v2_available and self.use_exllama:
            assert self.qweight.device.type == "cuda"
            assert self.qweight.device.index is not None
            self.q_tensors = {
                "qweight": self.qweight,
                "qzeros": self.qzeros,
                "scales": self.scales,
                "g_idx": self.g_idx,
            }
            temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
            self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (W[:, idx] + scale_zeros[self.g_idx[idx]])
                    / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
        )
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def unpack(self):
        with torch.no_grad():
            if self.wf.device != self.qzeros.device:
                self.wf = self.wf.to(self.qzeros.device)
            if self.bits in [2, 4, 8]:
                zeros = torch.bitwise_right_shift(
                    torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                    self.wf.unsqueeze(0),
                ).to(torch.int16 if self.bits == 8 else torch.int8)
                zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

                zeros = zeros + 1
                zeros = zeros.reshape(self.scales.shape)

                weight = torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                    self.wf.unsqueeze(-1),
                ).to(torch.int16 if self.bits == 8 else torch.int8)
                weight = torch.bitwise_and(weight, (2**self.bits) - 1)
            elif self.bits == 3:
                zeros = self.qzeros.reshape(
                    self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1
                ).expand(-1, -1, -1, 12)
                zeros = zeros >> self.wf.unsqueeze(0)
                zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
                    (zeros[:, :, 1, 0] << 2) & 0x4
                )
                zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
                    (zeros[:, :, 2, 0] << 1) & 0x6
                )
                zeros = zeros & 0x7
                zeros = torch.cat(
                    [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                    dim=2,
                )

                zeros = zeros + 1
                zeros = zeros.reshape(self.scales.shape)

                weight = self.qweight.reshape(
                    self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
                ).expand(-1, -1, 12, -1)
                weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
                weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | (
                    (weight[:, 1, 0] << 2) & 0x4
                )
                weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | (
                    (weight[:, 2, 0] << 1) & 0x6
                )
                weight = weight & 0x7
                weight = torch.cat(
                    [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
                )
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

            num_itr = 1
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

            linear = torch.nn.Linear(
                self.infeatures, self.outfeatures, bias=self.bias is not None
            )
            linear.weight = nn.Parameter(weights.t().float())
            if self.bias is not None:
                linear.bias = nn.Parameter(self.bias)

            return linear

    def forward(self, x: torch.Tensor):
        if self.bits == 4 and _exllama_v2_available and self.use_exllama:
            output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, False)
            if self.bias:
                output.add_(self.bias)
            return output

        elif self.bits == 3:
            out_shape = x.shape[:-1] + (self.outfeatures,)
            x = x.reshape(-1, x.shape[-1])
            if self.autogptq_cuda_available and (
                self.kernel_switch_threshold == 0
                or x.shape[0] < self.kernel_switch_threshold
            ):
                out = torch.zeros(
                    (x.shape[0], self.outfeatures), device=x.device, dtype=torch.float32
                )
                self.autogptq_cuda.vecquant3matmul(
                    x.float(),
                    self.qweight,
                    out,
                    self.scales.float(),
                    self.qzeros,
                    self.g_idx,
                )
            else:
                logger.warning(
                    f"Large kernel size {x.shape[0]}>{self.kernel_switch_threshold}, fallback to python implementation."
                )
                if self.wf.device != self.qzeros.device:
                    self.wf = self.wf.to(self.qzeros.device)
                zeros = self.qzeros.reshape(
                    self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1
                ).expand(-1, -1, -1, 12)
                zeros = zeros >> self.wf.unsqueeze(0)
                zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
                    (zeros[:, :, 1, 0] << 2) & 0x4
                )
                zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
                    (zeros[:, :, 2, 0] << 1) & 0x6
                )
                zeros = zeros & 0x7
                zeros = torch.cat(
                    [
                        zeros[:, :, 0, :11],
                        zeros[:, :, 1, 1:12],
                        zeros[:, :, 2, 1:11],
                    ],
                    dim=2,
                )

                zeros = zeros + 1
                zeros = zeros.reshape(self.scales.shape)

                weight = self.qweight.reshape(
                    self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
                ).expand(-1, -1, 12, -1)
                weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
                weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | (
                    (weight[:, 1, 0] << 2) & 0x4
                )
                weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | (
                    (weight[:, 2, 0] << 1) & 0x6
                )
                weight = weight & 0x7
                weight = torch.cat(
                    [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]],
                    dim=1,
                )
                weight = weight.reshape(
                    weight.shape[0] * weight.shape[1], weight.shape[2]
                )
                num_itr = self.g_idx.shape[0] // x.shape[-1]
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
                            scale_i[g_idx_i.long()]
                            * (weight_i - zeros_i[g_idx_i.long()])
                        )
                    weights = torch.cat(weights, dim=1)
                out = torch.matmul(x.half(), weights)
        else:
            out_shape = x.shape[:-1] + (self.outfeatures,)
            quant_linear_fn = QuantLinearInferenceOnlyFunction
            out = quant_linear_fn.apply(
                x.reshape(-1, x.shape[-1]),
                self.qweight,
                self.scales,
                self.qzeros,
                self.g_idx,
                self.bits,
                self.maxq,
            )
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        """
        Pre-tunes the quantized kernel
        """
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.outfeatures

            if (k, n) not in kn_values:
                kn_values[(k, n)] = (
                    m.qweight,
                    m.scales,
                    m.qzeros,
                    m.g_idx,
                    m.bits,
                    m.maxq,
                )

        logger.info(f"Found {len(kn_values)} unique KN Linear values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2**m
                for (k, n), (
                    qweight,
                    scales,
                    qzeros,
                    g_idx,
                    bits,
                    maxq,
                ) in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    quant_matmul_inference_only_248(
                        a, qweight, scales, qzeros, g_idx, bits, maxq
                    )
        del kn_values


__all__ = ["QuantLinear"]
