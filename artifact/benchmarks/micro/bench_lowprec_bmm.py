import torch
import triton.language as tl
import pandas as pd
from timeit import default_timer as timer
from fmzip.nn_modules.triton_utils.kernels import quant_matmul_248
from fmzip.nn_modules.triton_utils.bmm import quant_bmm_248
from fmzip.nn_modules.batched_qlinear import BatchedQuantLinearForward

tensors = torch.load(".cache/layer-1-selfattn.pt")

BATCH_SIZE = 64
BITS = 2
MAX_Q = 2**BITS - 1
timing_results = []

for batch_size in range(1, BATCH_SIZE + 1):
    input = torch.rand((batch_size, 1, 3200), dtype=torch.float16, device="cuda:0")
    projs = ["q_proj", "k_proj", "v_proj", "o_proj"]

    qweights = [tensors[f"model.layers.1.self_attn.q_proj.qweight"]] * batch_size
    qzeros = [tensors[f"model.layers.1.self_attn.q_proj.qzeros"]] * batch_size
    scales = [tensors[f"model.layers.1.self_attn.q_proj.scales"]] * batch_size
    g_idx = [tensors[f"model.layers.1.self_attn.q_proj.g_idx"]] * batch_size
    print(f"batch size={batch_size}")

    for iter in range(5):
        torch.cuda.synchronize()
        start = timer()
        results = []
        for i in range(batch_size):
            res = quant_matmul_248(
                input[i], qweights[i], scales[i], qzeros[i], g_idx[i], BITS, MAX_Q
            )
            results.append(res)
        non_batch_results = torch.stack(results)
        torch.cuda.synchronize()
        end = timer()
        if iter > 0:
            loop_time = end - start
        b_qzeros = torch.stack(qzeros)
        b_qweights = torch.stack(qweights)
        b_scales = torch.stack(scales)
        b_g_idx = torch.stack(g_idx)
        torch.cuda.synchronize()
        start = timer()
        batch_results = quant_bmm_248(
            input, b_qweights, b_scales, b_qzeros, b_g_idx, BITS, MAX_Q
        )
        torch.cuda.synchronize()
        end = timer()
        if iter > 0:
            timing_results.append(
                {"batch": end - start, "loop": loop_time, "batch_size": batch_size}
            )

        # for b in range(BATCH_SIZE):
        #     print(f"non_batch_output[{b}]={non_batch_results[b]}")
        #     print(f"batch_output[{b}]={batch_results[b]}")
        #     if torch.allclose(batch_results[b], non_batch_results[b], atol=1e-6, rtol=0):
        #         print("✅ match")
        #     else:
        #         print("❌ differ")

df = pd.DataFrame(timing_results)
df.to_csv("bench_lowprec_bmm.csv")
