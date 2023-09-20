import requests
from multiprocessing import Pool
endpoint = 'http://localhost:8000'

response_time = []
# rewrite the following with multi threading

def test(i):
    print(f"issuing {i}th request")
    task = {
        'prompt': "Once upon a time, ",
        'model': 'gpt2'
    }
    res = requests.post(endpoint + '/inference', json=task)
    return res.json()

with Pool(16) as p:
    results = p.map(test, range(16))
    print(results)