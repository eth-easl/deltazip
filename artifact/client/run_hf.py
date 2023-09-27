import os
import json
import requests
import subprocess
import signal
from multiprocessing import Pool
from timeit import default_timer as timer
import time

endpoint = 'http://localhost:8000'

def test(req):
    start = timer()
    res = requests.post(endpoint + '/inference', json=req)
    end = timer()
    return res.json(), end - start

with open('artifact/config.json', 'r') as fp:
    config = json.load(fp)

supported_models = config['supported_models']

test_prompt = "USER: Can you help me write a short essay about Alan Turing? ASSISTANT:"

test_data = [
    {
        'prompt': test_prompt,
        'model': x['hf_name']
    } for x in supported_models if x['type']=='finetuned'
]

proc = subprocess.Popen("FMZIP_BACKEND=hf FMZIP_BASE_MODEL=meta-llama/Llama-2-7b-hf uvicorn fmzip.rest.server:app", shell=True)
time.sleep(10)



with Pool(4) as p:
    results = p.map(test, test_data)
    print(results)

os.kill(proc.pid, signal.SIGKILL)