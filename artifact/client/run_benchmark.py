import json
import random
import requests
from loguru import logger
from multiprocessing import Pool
from timeit import default_timer as timer

endpoint = 'http://localhost:8000'

def inference_request(req):
    start = timer()
    res = requests.post(endpoint + '/inference', json=req)
    end = timer()
    print(res.json())
    print(end-start)
    logger.info("response received")
    return (res.json(), end - start)

def configure_server(backend: str, base_model: str, batch_size: int = 2):
    res = requests.post(endpoint+ "/restart", json={
        'backend': backend,
        'base_model': base_model,
        'batch_size': batch_size,
    })
    print("configuration finished...")
    return res.json()

with open('artifact/config.json', 'r') as fp:
    config = json.load(fp)

benchmark_results = []

supported_models = config['supported_models']
base_model = 'meta-llama/Llama-2-7b-hf'

test_prompt = "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:"

providers = ['hf', 'fmzip-mpm']

for provider in providers:
    if 'fmzip' in provider:
        test_data = [
            {
                'prompt': test_prompt,
                'model': x['compressed'],
            } for x in supported_models if x['type']=='finetuned'
        ]
    else:
        test_data = [
            {
                'prompt': test_prompt,
                'model': x['hf_name'],
            } for x in supported_models if x['type']=='finetuned'
        ]
    # randomly shuffle the test data
    random.shuffle(test_data)
    # step 1: config the server to use the provider
    configure_server(backend=provider, base_model=base_model)
    start = timer()
    with Pool(16) as p:
        results = p.map(inference_request, test_data)
    end = timer()
    benchmark_results.append({
        "provider": provider,
        "results": [
            {
                "id": x[0]['id'],
                "model": x[0]['model'],
                "prompt": x[0]['prompt'],
                "response": x[0]['response'],
                "time_elapsed": x[1],
            } for x in results
        ],
        "total_elapsed": end-start,
    })

with open('artifact/benchmark_results.json', 'w') as fp:
    json.dump(benchmark_results, fp, indent=2)