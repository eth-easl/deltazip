import torch
import numpy as np
import cupy as cp
from timeit import default_timer as timer
from src.lossless.compressor import LosslessCompressor

TENSOR_SIZE = 16384

# make a random tensor
x = torch.rand(TENSOR_SIZE, TENSOR_SIZE, device='cuda', dtype=torch.float16)
x_cpu_copy = x.cpu()

compressor = LosslessCompressor('gdeflate', 'fp16')

# compress
start = timer()
compressed_tensor, tensor_shape = compressor.compress_tensor(x)
end = timer()
print(f"Compressing {TENSOR_SIZE}x{TENSOR_SIZE} tensor takes {end - start} seconds")
# decompress
compressed_tensor = cp.array(compressed_tensor)
start = timer()

decompressed_tensor = compressor.decompress_tensor(compressed_tensor, tensor_shape)
end = timer()

# verify correctness
decompressed_tensor = decompressed_tensor.cpu()
assert torch.allclose(x_cpu_copy, decompressed_tensor, atol=1e-7)

print(f"Decompressing {TENSOR_SIZE}x{TENSOR_SIZE} tensor takes {end - start} seconds")
print(f"Decompress throughput: {TENSOR_SIZE * TENSOR_SIZE * 2 / (end - start) / 1024 / 1024 / 1024} GB/s")