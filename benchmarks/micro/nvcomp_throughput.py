import torch
import numpy as np
import cupy as cp
from timeit import default_timer as timer
from fmzip.lossless.compressor import LosslessCompressor

TENSOR_SIZE_X = 50304
TENSOR_SIZE_Y = 2560

# make a random tensor
x = torch.rand(TENSOR_SIZE_X, TENSOR_SIZE_Y, device='cuda', dtype=torch.float16)
x_cpu_copy = x.cpu()

compressor = LosslessCompressor('gdeflate')

# compress
start = timer()
compressed_tensor, tensor_shape = compressor.compress_tensor(x)
end = timer()
print(f"Compressing {TENSOR_SIZE_X}x{TENSOR_SIZE_Y} tensor takes {end - start} seconds")
# decompress
compressed_tensor = cp.array(compressed_tensor, dtype=cp.float16)
start = timer()

decompressed_tensor = compressor.decompress_tensor(compressed_tensor, tensor_shape)
end = timer()

# verify correctness
decompressed_tensor = decompressed_tensor.cpu()
assert torch.allclose(x_cpu_copy, decompressed_tensor, atol=1e-7)

print(f"Decompressing {TENSOR_SIZE_X}x{TENSOR_SIZE_Y} tensor takes {end - start} seconds")
print(f"Decompress throughput: {TENSOR_SIZE_X * TENSOR_SIZE_Y * 2 / (end - start) / 1024 / 1024 / 1024} GB/s")