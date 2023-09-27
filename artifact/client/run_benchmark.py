import os
import json
import requests
import subprocess
import signal
from multiprocessing import Pool
from timeit import default_timer as timer
import time
from loguru import logger
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
    return res.json()

with open('artifact/config.json', 'r') as fp:
    config = json.load(fp)

supported_models = config['supported_models']
base_model = 'meta-llama/Llama-2-7b-hf'

test_prompt = "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:"
# providers = ['hf', 'fmzip-mpm']
providers = ['hf']

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
    # step 1: config the server to use the provider
    configure_server(backend=provider, base_model=base_model)
    test_data = test_data[:1]
    print(test_data)
    with Pool(2) as p:
        results = p.map(inference_request, test_data)
    print(results)