import torch
from loguru import logger

none_tensor = torch.empty((1, 1), device="meta")

try:
    from exllamav2_kernels import make_q_matrix, gemm_half_q_half

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


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    # EXL2
    # won't work as the moment because the tensors are not the same.
    if "q_weight" in w:
        w["q_scale_max"] /= 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()
        return make_q_matrix(
            w["q_weight"],
            w["q_perm"],
            w["q_invperm"],
            w["q_scale"],
            w["q_scale_max"],
            w["q_groups"],
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )
    # GPTQ
    elif "qweight" in w:
        if w["scales"].dtype == torch.float:
            w["scales"] = w["scales"].half()

        # GPTQ with g_idx (act_order)
        if "g_idx" in w and not (w["g_idx"] == 0).all().item():
            w["q_perm"] = torch.empty(
                (w["qweight"].shape[0] * 8,),
                dtype=torch.short,
                device=w["qweight"].device,
            )
            w["q_invperm"] = torch.empty_like(w["q_perm"])
            # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
            return make_q_matrix(
                w["qweight"],
                w["q_perm"],
                w["q_invperm"],
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                w["g_idx"].cpu(),
                temp_dq,
            )
        # GPTQ without g_idx
        else:
            return make_q_matrix(
                w["qweight"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                none_tensor,
                temp_dq,
            )


class ExLlamaV2DeviceTensors:
    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2,),
            dtype=torch.half,
            device=_torch_device(self.device_idx),
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
