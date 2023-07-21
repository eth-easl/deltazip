import torch
from timeit import default_timer as timer
from src.lossless.compressor import LosslessCompressor
TENSOR_SIZE = 2048

# make a random tensor
x = torch.rand(TENSOR_SIZE, TENSOR_SIZE, TENSOR_SIZE, device='cuda', dtype=torch.float16)

# warm up
compressor = LosslessCompressor('gdeflate', 'fp16')

# compress
start = timer()
compressed_tensor, tensor_shape = compressor.compress_tensor(x)
end = timer()
print(f"Compressing {TENSOR_SIZE}x{TENSOR_SIZE}x{TENSOR_SIZE} tensor takes {end - start} seconds")

# decompress
start = timer()
decompressed_tensor = compressor.decompress_tensor(compressed_tensor, tensor_shape)
end = timer()
print(f"Decompressing {TENSOR_SIZE}x{TENSOR_SIZE}x{TENSOR_SIZE} tensor takes {end - start} seconds")
print(f"Decompress throughput: {TENSOR_SIZE * TENSOR_SIZE * TENSOR_SIZE * 2 / (end - start) / 1024 / 1024 / 1024} GB/s")