import os
import json
from tabulate import tabulate

def get_sysname(meta_info):
    sys_info = meta_info['sys_info']
    if 'delta_modules' in sys_info and len(sys_info['delta_modules']) > 0:
        return f"DeltaZip ({sys_info['max_deltas']})"
    elif 'swap_modules' in sys_info and len(sys_info['swap_modules']) > 0:
        return f"Baseline ({sys_info['max_swap_slots']})"
    else:
        raise ValueError("Unknown system")

def calculate_average_latency(data):
    data = [x['response'] for x in data]
    latencies = [x['metrics'][0]['finished_time'] - x['metrics'][0]['arrival_time'] for x in data]
    return sum(latencies) / len(latencies)

def calculate_ttft(data):
    data = [x['response'] for x in data]
    latencies = [x['metrics'][0]['first_token_time'] - x['metrics'][0]['arrival_time'] for x in data]
    return sum(latencies) / len(latencies)

def calculate_throughput(data):
    data = [x['response'] for x in data]
    max_finished_time = max([x['metrics'][0]['finished_time'] for x in data])
    min_arrival_time = min([x['metrics'][0]['arrival_time'] for x in data])
    total_time = max_finished_time - min_arrival_time
    return len(data) / total_time

def aggregate_perf(args):
    results = [x for x in os.listdir(args.dir) if x.endswith('.jsonl')]
    perfs = []
    for res in results:
        with open(os.path.join(args.dir, res)) as f:
            data = [json.loads(line) for line in f]
            meta_info, data = data[0], data[1:]
            sysname = get_sysname(meta_info)
            avg_latency = calculate_average_latency(data)
            avg_ttft = calculate_ttft(data)
            throughput = calculate_throughput(data)
            perfs.append({
                'sysname': sysname,
                'avg_latency': f"{avg_latency:.2f}",
                'avg_ttft': f"{avg_ttft:.2f}",
                'throughput': f"{throughput:.4f}"
            })
    table = tabulate(perfs, headers='keys', tablefmt='pretty')

    print(table)
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate performance results')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing performance results')
    args = parser.parse_args()
    aggregate_perf(args)