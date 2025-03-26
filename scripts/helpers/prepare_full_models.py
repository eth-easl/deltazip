import os
from tqdm import tqdm

def prepare(args):
    print(args)
    num_copies = args.num_copies
    print(f"Downloading {args.model} to {args.target}")
    folder_name = args.model.split("/")[-1]
    os.makedirs(args.target, exist_ok=True)
    cmd_download_first = f"huggingface-cli download {args.model} --local-dir {args.target}/{folder_name}.0"
    os.system(cmd_download_first)
    for i in tqdm(range(num_copies-1)):
        copy_cmd = f"cp -r {args.target}/{folder_name}.0 {args.target}/{folder_name}.{i+1}"
        os.system(copy_cmd)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--num-copies", type=int, default=4)
    prepare(parser.parse_args())