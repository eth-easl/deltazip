import torch
import numpy as np
import cupy as cp
from timeit import default_timer as timer
from deltazip.lossless.compressor import LosslessCompressor

TENSOR_SIZE_X = 32768
TENSOR_SIZE_Y = 16384

# make a random tensor
x = torch.rand(TENSOR_SIZE_X, TENSOR_SIZE_Y, device="cuda", dtype=torch.float16)
x_cpu_copy = x.cpu()

compressor = LosslessCompressor("gdeflate")

# compress
start = timer()
compressed_tensor, tensor_shape = compressor.compress_tensor(x)
print(compressed_tensor)
print(tensor_shape)
end = timer()
print(f"Compressing {TENSOR_SIZE_X}x{TENSOR_SIZE_Y} tensor takes {end - start} seconds")


# decompress
compressed_tensor = cp.array(compressed_tensor)
print(compressed_tensor.shape)
start = timer()

decompressed_tensor = compressor.decompress_tensor(compressed_tensor, tensor_shape)
end = timer()

# verify correctness
decompressed_tensor = decompressed_tensor.cpu()
assert torch.allclose(x_cpu_copy, decompressed_tensor, atol=1e-7)

print(
    f"Decompressing {TENSOR_SIZE_X}x{TENSOR_SIZE_Y} tensor takes {end - start} seconds"
)
print(
    f"Decompress throughput: {TENSOR_SIZE_X * TENSOR_SIZE_Y * 2 / (end - start) / 1024 / 1024 / 1024} GB/s"
)
