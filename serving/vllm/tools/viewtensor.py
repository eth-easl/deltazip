import safetensors as st


def main(args):
    print(args)
    tensor_stats = {}
    with st.safe_open(args.ckpt, "pt") as f:
        for key in f.keys():
            temp_tensor = f.get_tensor(key)
            tensor_stats[key] = {
                "size": temp_tensor.shape,
                "dtype": temp_tensor.dtype,
                "mean": temp_tensor.float().mean().item(),
                "std": temp_tensor.float().std().item(),
            }
    for key, value in tensor_stats.items():
        print(
            f"key: {key} | size: {value['size']} | dtype: {value['dtype']} | mean: {value['mean']} | std: {value['std']}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View tensor")
    parser.add_argument("--ckpt", type=str, help="Path to tensor")

    args = parser.parse_args()
    main(args)
