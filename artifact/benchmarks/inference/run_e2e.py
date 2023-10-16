import os
import json
import argparse
import requests
import threading
import numpy as np
from loguru import logger
from timeit import default_timer as timer
import sched, time

endpoint = "http://localhost:8000"
inference_results = []
s = sched.scheduler(time.monotonic, time.sleep)
threads = []

def request_thread(req):
    global inference_results
    print(f"issuing... {req}")
    res = requests.post(endpoint + "/inference", json=req)
    res = {
        "response": res.json(),
    }
    print(f"received... {res}")
    inference_results.append(res)
    return res

def async_issue_requests(reqs):
    
    global threads
    for req in reqs:
        print(f"req: {req}")
        thread = threading.Thread(target=request_thread, args=(req,))
        threads.append(thread)
        thread.start()

def issue_queries(queries):
    time_step = 0.01
    global threads
    global inference_results
    time_range = [x["timestamp"] for x in queries]
    max_time = max(time_range) + 1  # execute for one more second
    start = timer()
    for time in np.arange(0, max_time, time_step):
        sub_queries = [
            x
            for x in queries
            if x["timestamp"] <= time and x["timestamp"] > time - time_step
        ]
        if len(sub_queries) > 0:
            s.enter(time, 1, async_issue_requests, argument=(sub_queries,))
    s.run(blocking=True)
    print(f"total threads: {len(threads)}")
    [thread.join() for thread in threads]
    end = timer()
    logger.info("all queries issued")
    return {"results": inference_results, "total_elapsed": end - start}


def configure_system(backend, base_model, backend_args, model_mapping, gen_configs):
    res = requests.post(
        endpoint + "/restart",
        json={
            "backend": backend,
            "base_model": base_model,
            "backend_args": backend_args,
            "mapping": model_mapping,
            "gen_configs": gen_configs,
        },
    )
    logger.info("configuration finished...")


def run(args):
    systems = [
        x
        for x in os.listdir(args.workload)
        if x.endswith(".json") and x.startswith("system")
    ]
    with open(os.path.join(args.workload, "trace.json"), "r") as fp:
        jobs = json.load(fp)
    with open(os.path.join(args.workload, "config.json"), "r") as fp:
        config = json.load(fp)
    base_model = config["base_model"]
    model_mapping = config["compressed_model_mapping"]
    gen_configs = config["generation_configs"]
    print(f"gen configs: {gen_configs}")
    # order systems, in desending order, by name
    systems = sorted(systems, key=lambda x: x, reverse=True)
    for system in systems:
        with open(os.path.join(args.workload, system), "r") as fp:
            sys_config = json.load(fp)["systems"]
        configure_system(
            backend=sys_config[0]["name"],
            base_model=base_model,
            backend_args=sys_config[0]["args"],
            model_mapping=model_mapping,
            gen_configs=gen_configs,
        )
        issue_queries(jobs["queries"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    run(args)
