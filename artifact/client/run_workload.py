import json
import requests
import argparse
from loguru import logger
from timeit import default_timer as timer

endpoint = "http://localhost:8000"

def inference_request(req):
    start = timer()
    res = requests.post(endpoint + "/inference", json=req)
    end = timer()
    print(res.json())
    print(end - start)
    logger.info("response received")
    return (res.json(), end - start)

def configure_server(backend: str, base_model: str, batch_size: int = 1, model_parallel_strategy="none"):
    res = requests.post(
        endpoint + "/restart",
        json={
            "backend": backend,
            "base_model": base_model,
            "batch_size": batch_size,
            "model_parallel_strategy": model_parallel_strategy,
        },
    )
    logger.info("configuration finished...")
    return res.json()

def main(workload_file: str):
    with open(workload_file, "r") as fp:
        workload = json.load(fp)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-file", type=str, default="test.json")
    parser.add_argument("--output-file", type=str, default="test.out.json")
    args = parser.parse_args()