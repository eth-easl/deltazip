from argparse import ArgumentParser

def load_naive(args):
    return args.model_size * args.dtype / args.disk_gpu_bandwidth

def load_compressed(args):
    load_size =  args.model_size * args.dtype / args.compression_rate
    load_time = load_size / args.disk_gpu_bandwidth
    decompress_time = load_size / args.decompression_throughput
    return load_time + decompress_time + args.add_back_overhead

if __name__=="__main__":
    # DISK_GPU_BANDWIDTH = 3.5 # GB / s
    # DECOMPRESSION_THROUGHPUT = 45 # GB / s
    # MODEL_SIZE = 6.7 # in billion parameters
    # DTYPE = 2 # in # bytes per parameter
    # ADD_BACK_OVERHEAD = 2 # in seconds
    parser = ArgumentParser()
    parser.add_argument("--compression-rate", type=float, required=True)
    parser.add_argument("--model-sizes", type=float, required=True, nargs="+")
    parser.add_argument("--dtype", type=float, required=True, default=2, choices=[2, 4])
    parser.add_argument("--decompression-throughput", type=float, required=True)
    parser.add_argument("--disk-gpu-bandwidth", type=float, required=True)
    parser.add_argument("--add-back-overhead", type=float, required=True, default=2)
    args = parser.parse_args()
    for model_size in args.model_sizes:
        args.model_size = model_size
        print("Model size: {}B".format(model_size))
        naive_load_time = load_naive(args)
        print("Naive loading time: {}s".format(naive_load_time))
        compressed_load_time = load_compressed(args)
        print("Compressed loading time: {}s".format(compressed_load_time))
