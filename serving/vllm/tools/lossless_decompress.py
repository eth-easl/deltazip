from vllm.delta.compressor import LosslessCompressor
import safetensors as st
import json
import cupy as cp
from safetensors.torch import save_file


def main(args):
    print(args)
    tensors = {}
    with st.safe_open(args.ckpt, framework="torch", device="cuda:0") as f:
        metadata = f.metadata()
        keys = f.keys()
        for key in keys:
            tensors[key] = f.get_tensor(key)
        tensor_dtypes = json.loads(metadata["dtype"])
        tensor_shapes = json.loads(metadata["shape"])

    with cp.cuda.Device(0):
        for key in tensors.keys():
            tensors[key] = cp.array(tensors[key], copy=False)
    lc = LosslessCompressor()
    print("decompression starts")
    tensors = lc.decompress_state_dict(
        tensors,
        tensor_shapes,
        tensor_dtypes,
        use_bfloat16=False,
        target_device="cuda:0",
    )
    # save the decompressed tensors
    save_file(
        tensors,
        args.ckpt,
    )
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to the compressed file")
    args = parser.parse_args()
    main(args)
