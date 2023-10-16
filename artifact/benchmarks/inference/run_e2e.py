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

def async_issue_requests(req):
    results = []
    start = timer()
    global inference_results
    res = requests.post(endpoint + "/inference", json=req)
    res = {
        "response": res.json(),
    }
    end = timer()
    res["time_elapsed"] = end - start
    logger.info(f"query {req['id']} elapsed {res['time_elapsed']} seconds")
    inference_results.append(res)
    return results

def issue_queries(queries):
    time_step = 0.01
    global inference_results
    inference_results = []
    # first find the range of the timestamp
    time_range = [x["timestamp"] for x in queries]
    max_time = max(time_range) + 1  # execute for one more second
    # with a step 0.01
    threads = []
    start = timer()

    for time in np.arange(0, max_time, time_step):
        # find all queries that are within this time range
        # and issue them
        sub_queries = [
            x
            for x in queries
            if x["timestamp"] <= time and x["timestamp"] > time - time_step
        ]
        if len(sub_queries) > 0:
            s.enter(time, 1, async_issue_requests, argument=(sub_queries,))
    s.run(blocking=False)
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
    jobs = [
        x
        for x in os.listdir(args.workload)
        if x.endswith(".json") and x.startswith("trace")
    ]
    with open(os.path.join(args.workload, "config.json"), "r") as fp:
        config = json.load(fp)
    base_model = config["base_model"]
    model_mapping = config["compressed_model_mapping"]
    gen_configs = config["generation_configs"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    run(args)
