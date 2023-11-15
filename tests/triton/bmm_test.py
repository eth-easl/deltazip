import torch
import triton
import triton.language as tl
from fmzip.nn_modules.triton_utils.bmm import quant_bmm_248
from fmzip.nn_modules.triton_utils.kernels import quant_matmul_inference_only_248

tensors = torch.load(".cache/2bits-openllama/layer-1-selfattn.pt")
BATCH_SIZE = 4
BITS = 2
MAX_Q = 2 ** BITS - 1

input = torch.rand((BATCH_SIZE, 256, 3200), dtype=torch.float16, device="cuda:0")
projs = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

qweights = [tensors[f"model.layers.1.self_attn.{proj}.qweight"] for proj in projs][:BATCH_SIZE]
qzeros = [tensors[f"model.layers.1.self_attn.{proj}.qzeros"] for proj in projs][:BATCH_SIZE]
scales = [tensors[f"model.layers.1.self_attn.{proj}.scales"] for proj in projs][:BATCH_SIZE]
g_idx = [tensors[f"model.layers.1.self_attn.{proj}.g_idx"] for proj in projs][:BATCH_SIZE]

results = []
for i in range(BATCH_SIZE):
    res = quant_matmul_inference_only_248(input[i], qweights[i], scales[i], qzeros[i], g_idx[i], BITS, MAX_Q)
    results.append(res)
non_batch_results = torch.stack(results)
print(non_batch_results.shape)

qweights = torch.stack(qweights)
qzeros = torch.stack(qzeros)
scales = torch.stack(scales)
g_idx = torch.stack(g_idx)
batch_results = quant_bmm_248(input, qweights, scales, qzeros, g_idx, BITS, MAX_Q)

for b in range(BATCH_SIZE):
    print(f"non_batch_output[{b}]={non_batch_results[b]}")
    print(f"batch_output[{b}]={batch_results[b]}")
    if torch.allclose(batch_results[b], non_batch_results[b], atol=1e-6, rtol=0):
        print("✅ match")
    else:
        print("❌ differ")