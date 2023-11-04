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


def request_thread(
    req,
    start_time,
    global_start_time,
):
    global inference_results
    res = requests.post(endpoint + "/inference", json=req)
    end_time = timer()
    res = {
        "response": res.json(),
        "time_elapsed": end_time - start_time,
        "relative_start_at": start_time - global_start_time,
    }
    inference_results.append(res)
    return res


def async_issue_requests(reqs, global_start_time):
    global threads
    for req in reqs:
        start_time = timer()
        thread = threading.Thread(
            target=request_thread,
            args=(
                req,
                start_time,
                global_start_time,
            ),
        )
        threads.append(thread)
        thread.start()


def issue_queries(queries):
    time_step = 0.1
    global threads
    global inference_results
    time_range = [x["timestamp"] for x in queries]
    max_time = max(time_range) + 1  
    # execute for one more second
    start = timer()
    for time in np.arange(0, max_time, time_step):
        sub_queries = [
            x
            for x in queries
            if x["timestamp"] <= time and x["timestamp"] > time - time_step
        ]
        if len(sub_queries) > 0:
            print(f"sending {len(sub_queries)} queries at {time}")
            s.enter(
                time,
                1,
                async_issue_requests,
                argument=(
                    sub_queries,
                    start,
                ),
            )

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
    global inference_results
    with open(os.path.join(args.workload, "trace.json"), "r") as fp:
        jobs = json.load(fp)
    with open(os.path.join(args.workload, "config.json"), "r") as fp:
        config = json.load(fp)
    base_model = config["base_model"]
    model_mapping = config["compressed_model_mapping"]
    gen_configs = config["generation_configs"]
    print(f"gen configs: {gen_configs}")
    benchmark_results = []
    # order systems, in desending order, by name
    benched_systems = []
    for system in systems:
        with open(os.path.join(args.workload, system), "r") as fp:
            sys_config = json.load(fp)["systems"]
        benched_systems.append(sys_config)
    benched_systems = sorted(benched_systems, key=lambda x: x[0]["order"])
    for sys_config in benched_systems:
        configure_system(
            backend=sys_config[0]["name"],
            base_model=base_model,
            backend_args=sys_config[0]["args"],
            model_mapping=model_mapping,
            gen_configs=gen_configs,
        )
        print("Start sending requests...")
        issue_queries(jobs["queries"])
        benchmark_results.append(
            {
                "system": sys_config[0],
                "gen_configs": gen_configs,
                "results": inference_results,
            }
        )
        inference_results = []
    output = args.workload.replace("workloads", "results") + ".json"
    # create dir if not exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as fp:
        json.dump(benchmark_results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="")
    args = parser.parse_args()
    run(args)
