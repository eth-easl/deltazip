from deltazip.utils.generate import generate

def main(args):
    print(args)
    generate(args.target_model, args.prompt)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--prompt", type=str, default="Alan Turing is ")
    main(parser.parse_args())