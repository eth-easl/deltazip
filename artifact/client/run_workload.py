import argparse


def parse_workload():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--workload-file", type=str, default="test.json")
    parser.add_argument("--output-file", type=str, default="test.out.json")
