import json
import cupy as cp
from safetensors.numpy import safe_open
from fmzip.lossless.compressor import LosslessCompressor

losslesscompressor = LosslessCompressor(algorithm='gdeflate', device_id=0)

tensors = {}

with safe_open('.cache/compressed_models/3b-parameters/openllama-chat-1/fmzip-compressed.safetensors', framework="numpy") as f:
    metadata = f.metadata()
    keys = f.keys()
    for key in keys:
        tensors[key] = f.get_tensor(key)

tensor_dtypes = json.loads(metadata["dtype"])
tensor_shapes = json.loads(metadata["shape"])
# (todo: xiaozhe), (todo: minor)
# seems like we cannot use arbitrary device to decompress
# for now use device=0 to decompress and then move to target device
with cp.cuda.Device(0):
    for key in tensors.keys():
        tensors[key] = cp.array(tensors[key], copy=False)
for key in tensors.keys():
    print(tensors[key].device)
    break

tensors = losslesscompressor.decompress_state_dict(
    tensors, tensor_shapes, tensor_dtypes, use_bfloat16=False
)

print(tensors.device)