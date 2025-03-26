import os
from tqdm import tqdm

def copy(args):
    print(args)
    print(f"Copying {args.source} to {args.target}")
    num_copies = args.num_copies
    folder_name = args.source.split("/")[-1]
    os.makedirs(args.target, exist_ok=True)
    for i in tqdm(range(num_copies)):
        os.system(f"cp -r {args.source} {args.target}/{folder_name}.{i}")
    print(f"Done copying {args.source} to {args.target}")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--num-copies", type=int, default=24)
    copy(parser.parse_args())
