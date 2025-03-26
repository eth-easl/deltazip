import os
import json

color_palette = {
    "general": [
        "#90a0c8",
        "#f19e7b",
        "#72ba9d",
        "#bfc8c9",
        "#f9daad",
        "#fbe9d8",
    ]
}

def merge_sysinfo(sysinfo):
    endpoints = list(sysinfo.keys())
    # merge principles: if the keys exist, skip
    # otherwise, add
    # print(sysinfo)
    first_sysinfo = sysinfo
    # first_sysinfo = sysinfo[endpoints[0]]
    first_sysinfo['lora_modules'] = [sysinfo[x]['lora_modules'] for x in endpoints if len(sysinfo[x]['lora_modules']) > 0]
    first_sysinfo['delta_modules'] = [sysinfo[x]['delta_modules'] for x in endpoints if len(sysinfo[x]['delta_modules']) > 0]
    first_sysinfo['swap_modules'] = [sysinfo[x]['swap_modules'] for x in endpoints if len(sysinfo[x]['swap_modules']) > 0]

    if len(first_sysinfo['lora_modules']) > 0:
        first_sysinfo['lora_modules'] = first_sysinfo['lora_modules'][0]

    if len(first_sysinfo['delta_modules']) > 0:
        first_sysinfo['delta_modules'] = first_sysinfo['delta_modules'][0]

    
    if len(first_sysinfo['swap_modules']) > 0:
        first_sysinfo['swap_modules'] = first_sysinfo['swap_modules'][0]
    
    return first_sysinfo

def get_system_name(sys):
    sys = sys.lower()
    if "vllm" in sys:
        return "Baseline-1"
    if "deltaserve" in sys:
        if "prefetch" not in sys:
            return "Ours"
        else:
            return "Ours+"
    return "Unknown"


system_color_mapping = {"Baseline-1": "#90a0c8", "Ours": "#f19e7b", "Ours+": "#72ba9d"}


def parse_annotations(annotations: str):
    """annotations are in format: key1=val1,key2=val2,...
    this function parse it into dictionary as {key1: val1, key2: val2, ...}
    """
    pairs = annotations.split(",")
    parsed = {}
    for pair in pairs:
        key, val = pair.split("=")
        parsed[key] = val
    return parsed


def extract_key_metadata(metadata):
    workload = parse_annotations(
        metadata["workload"].split("/")[-1].removesuffix(".jsonl")
    )
    gen_tokens = metadata["workload"].split("/")[-2].removeprefix("gen_")
    endpoints = list(metadata["sys_info"].keys())
    # for old log files, comment this line below
    # metadata["sys_info"] = merge_sysinfo(metadata["sys_info"])
    
    tp_size = metadata["sys_info"]["tensor_parallel_size"]
    is_swap = len(metadata["sys_info"]["swap_modules"]) > 0
    is_delta = len(metadata["sys_info"]["delta_modules"]) > 0
    is_lora = len(metadata["sys_info"]["lora_modules"]) > 0
    total_models = len(metadata["sys_info"]["delta_modules"]) + len(
        metadata["sys_info"]["swap_modules"]
    )
    max_deltas = metadata["sys_info"]["max_deltas"]
    max_swaps = metadata["sys_info"]["max_swap_slots"]
    max_cpu_swaps = metadata["sys_info"]["max_cpu_models"]
    max_cpu_deltas = metadata["sys_info"]["max_cpu_deltas"]

    if is_delta:
        total_models = total_models + 1
    enable_prefetch = True
    bitwidth = 4
    if "enable_prefetch" in metadata["sys_info"]:
        if not metadata["sys_info"]["enable_prefetch"]:
            enable_prefetch = False
    is_nvme = False
    if is_swap:
        bitwidth = 16
        if metadata["sys_info"]["swap_modules"][0]["local_path"].startswith("/scratch"):
            is_nvme = True
    if is_delta:
        if metadata["sys_info"]["delta_modules"][0]["local_path"].startswith(
            "/scratch"
        ):
            is_nvme = True
        if "2b" in metadata["sys_info"]["delta_modules"][0]["local_path"]:
            bitwidth = 2
        if "4b" in metadata["sys_info"]["delta_modules"][0]["local_path"]:
            bitwidth = 4
    is_unoptimized_delta = False
    policy = "fcfs"
    if "scheduler_policy" in metadata["sys_info"]:
        policy = metadata["sys_info"]["scheduler_policy"]
    if is_delta:
        if "unopt" in metadata["sys_info"]["delta_modules"][0]["local_path"]:
            is_unoptimized_delta = True

    workload.update(
        {
            "max_deltas": max_deltas,
            "max_swaps": max_swaps,
            "max_cpu_swaps": max_cpu_swaps,
            "max_cpu_deltas": max_cpu_deltas,
            "bitwidth": bitwidth,
            "tp_size": tp_size,
            "is_swap": is_swap,
            "is_delta": is_delta,
            "is_lora": is_lora,
            "is_unoptimized_delta": is_unoptimized_delta,
            "gen_tokens": gen_tokens,
            "is_nvme": is_nvme,
            "enable_prefetch": enable_prefetch,
            "policy": policy,
            "total_models": total_models,
        }
    )
    return workload


def _parse_data(data):
    results = []
    for id, x in enumerate(data):
        metric = x["response"]["metrics"][0]
        e2e_latency = metric["finished_time"] - metric["arrival_time"]
        first_token_latency = metric["first_token_time"] - metric["arrival_time"]
        queuing_time = metric["first_scheduled_time"] - metric["arrival_time"]
        if metric['cpu_loading_time'] is None and metric['gpu_loading_time'] is None:
            gpu_loading_time = 0
            cpu_loading_time = 0
        elif metric['cpu_loading_time'] is None:
            raise ValueError("CPU loading time is None but GPU loading time is not None")
        elif metric['gpu_loading_time'] is None:
            raise ValueError("GPU loading time is None but CPU loading time is not None")
        else:
            gpu_loading_time = metric["gpu_loading_time"] - metric["cpu_loading_time"]
            cpu_loading_time = metric["cpu_loading_time"] - metric["first_scheduled_time"]
            
        inference_time = metric["finished_time"] - gpu_loading_time

        arrival_time = metric["arrival_time"]
        finish_time = metric["finished_time"]
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": e2e_latency,
                "type": "E2E Latency",
                "arrival_time": arrival_time,
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": first_token_latency,
                "type": "TTFT",
                "arrival_time": arrival_time,
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": gpu_loading_time + cpu_loading_time,
                "type": "Loading",
                "arrival_time": arrival_time,
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": inference_time,
                "type": "Inference",
                "arrival_time": arrival_time,
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": queuing_time,
                "type": "Queueing",
                "arrival_time": arrival_time,
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": arrival_time,
                "type": "Arrival",
                "arrival_time": arrival_time,
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": finish_time,
                "type": "Finish",
                "arrival_time": arrival_time,
            }
        )
    return results


def _parse_data_order(data):
    results = []
    for id, x in enumerate(data):
        metric = x["response"]["metrics"][0]
        model = x["response"]["model"]
        arrival_time = metric["arrival_time"]
        result = {
                "id": id,
                "model": model,
                "arrival": arrival_time,
                "queueing_start": arrival_time,
                "queueing_end": metric["first_scheduled_time"],
                "loading_start": metric["first_scheduled_time"],
                "loading_end": metric["gpu_loading_time"],
                "first_token_start": metric["gpu_loading_time"],
                "first_token_end": metric["first_token_time"],
                "inference_start": metric["first_token_time"],
                "inference_end": metric["finished_time"],
                "E2E Latency": metric["finished_time"] - arrival_time,
                "TTFT": metric["first_token_time"] - arrival_time,
            }
        if len(metric["preempty_in_times"]) >0 or len(metric["preempty_out_times"]) > 0:
            for i, time in enumerate(metric["preempty_in_times"]):
                result[f"preempt_in_{i}"] = time
            for i, time in enumerate(metric["preempty_out_times"]):
                result[f"preempt_out_{i}"] = time
        results.append(result)        
    return results


def parse_data(input_file, order=False):
    with open(input_file, "r") as fp:
        data = [json.loads(line) for line in fp]
    metadata = data.pop(0)
    key_metadata = extract_key_metadata(metadata)
    if order:
        results = _parse_data_order(data)
    else:
        results = _parse_data(data)
    return key_metadata, results


def get_title(key_metadata):
    sys = "Unknown"
    hardware = "Unknown"
    if key_metadata["is_swap"]:
        sys = "\\text{vLLM}"
    if key_metadata["is_delta"]:
        sys = "\\text{DeltaServe}"
        sys += f"({key_metadata['bitwidth']}bit)"
        if key_metadata["is_unoptimized_delta"]:
            pass
        if key_metadata["is_delta"] and not key_metadata["is_unoptimized_delta"]:
            sys += "\\text{+I/O}"
        if key_metadata["enable_prefetch"]:
            sys += "\\text{+Prefetch}"
    workload = ""
    # workload = "\\text{<>}, ".replace("<>", key_metadata["distribution"])
    if "ar" in key_metadata:
        workload += f"\lambda={key_metadata['ar']}"
    if key_metadata["is_nvme"]:
        hardware = "\\text{NVMe}"
    else:
        hardware = "\\text{NFS}"
    sys = "\Large{" + sys + "," + key_metadata["policy"] + "}"
    workload = "\Large{" + workload + "}"
    hardware = "\Large{" + hardware + "}"
    return f"${sys}, {workload}, {hardware}$"


def get_sys_name(key_metadata):
    sys = "Unknown"
    hardware = "Unknown"
    if key_metadata["is_swap"]:
        sys = "\\text{vLLM}"
    if key_metadata["is_delta"]:
        sys = "\\text{DeltaServe}"
        sys += str(key_metadata["bitwidth"]) + "\\text{bit}"
        if key_metadata["is_unoptimized_delta"]:
            pass
        if key_metadata["is_delta"] and not key_metadata["is_unoptimized_delta"]:
            sys += "\\text{+I/O}"
        if key_metadata["enable_prefetch"]:
            sys += "\\text{+Prefetch}"
    if key_metadata["is_nvme"]:
        hardware = "\\text{NVMe}"
    else:
        hardware = "\\text{NFS}"
    sys = "\Large{" + sys + "}"
    hardware = "\Large{" + hardware + "}"
    return f"${sys}, {hardware}$"


def get_short_system_name(key_metadata):
    if key_metadata["is_swap"]:
        return "Baseline-1", 0
    if (
        key_metadata["is_delta"]
        and key_metadata["enable_prefetch"]
        and key_metadata["policy"] == "deltaserve"
    ):
        return "+Policy", 3
    elif key_metadata["is_delta"] and key_metadata["enable_prefetch"]:
        return "+Prefetch", 2
    elif key_metadata["is_delta"]:
        return "+Delta", 1
    elif key_metadata["is_lora"]:
        return "LoRA", 0
    else:
        raise ValueError("Unknown system type")

def get_mixed_system_name(key_metadata):
    if key_metadata["is_swap"] and key_metadata["is_lora"]:
        return "Swap + LoRA", 0
    if key_metadata["is_delta"] and key_metadata["is_lora"]:
        return "Delta + LoRA", 1
    

def walk_through_files(path, file_extension=".jsonl"):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dirpath, filename)
