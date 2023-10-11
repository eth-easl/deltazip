import json
import requests
import argparse
import threading
import numpy as np
from loguru import logger
from multiprocessing import Pool
from timeit import default_timer as timer
from copy import deepcopy

endpoint = "http://localhost:8000"

def inference_request(req):
    start = timer()
    res = requests.post(endpoint + "/inference", json=req)
    end = timer()
    print(res.json())
    print(end - start)
    logger.info("response received")
    return (res.json(), end - start)

def parallel_issue_requests(reqs):
    with Pool(len(reqs)) as p:
        results = p.map(inference_request, reqs)
    return results

def configure_server(backend: str, base_model: str, batch_size: int = 1, model_parallel_strategy="none"):
    logger.info(f"configuring server with backend=[{backend}], base model=[{base_model}], batch size=[{batch_size}], model parallel strategy=[{model_parallel_strategy}]")

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

def issue_queries(queries, backend, mapping=None):
    time_step = 0.01
    reformatted_queries = deepcopy(queries)
    if backend['mapping']:
        for i, query in enumerate(queries):
            reformatted_queries[i]['model'] = mapping[query['model']]
    for idx, query in enumerate(reformatted_queries):
        reformatted_queries[idx]['id'] = str(idx)
    # first find the range of the timestamp
    time_range = [x['timestamp'] for x in queries]
    max_time = max(time_range) + 1 # execute for one more second
    # with a step 0.01
    threads = []
    for time in np.arange(0, max_time, time_step):
        # find all queries that are within this time range
        # and issue them
        sub_queries = [x for x in reformatted_queries if x['timestamp'] <= time and x['timestamp'] > time - time_step]
        if len(sub_queries) > 0:
            logger.info(f"# of queries to be issued: {len(sub_queries)}")
            # start a new non-blocking process to issue these queries
            thread = threading.Thread(target=parallel_issue_requests, args=(sub_queries,))
            threads.append(thread)
            thread.start()

    [thread.join() for thread in threads]
    logger.info("all queries issued")

def main(args):
    print(args)
    with open(args.workload, "r") as fp:
        workload = json.load(fp)
    backends = workload['systems']
    model_mapping = workload['compressed_model_mapping']
    for backend in backends:
        configure_server(
            backend=backend['name'],
            base_model=workload['base_model'],
            batch_size=backend['args'].get('batch_size', 1),
            model_parallel_strategy=backend['args'].get('model_parallel_strategy', 'none')
        )
        issue_queries(workload['queries'], backend, model_mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="test.json")
    parser.add_argument("--output-file", type=str, default="test.out.json")
    args = parser.parse_args()
    main(args)