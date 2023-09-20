import requests
from multiprocessing import Pool
from timeit import default_timer as timer

endpoint = 'http://localhost:8000'

response_time = []
task = {
    'prompt': "Once upon a time, ",
    'model': '.cache/compressed_models/p2.8b_gsd_133'
}

# task = {
#     'prompt': "Once upon a time, ",
#     'model': '.cache/raw_models/gsd/step_133'
# }

def test(i):
    print(f"issuing {i}th request")
    res = requests.post(endpoint + '/inference', json=task)
    return res.json()

start = timer()
with Pool(4) as p:
    results = p.map(test, range(4))
    print(results)
end = timer()
print(f"Total time: {end - start} s")