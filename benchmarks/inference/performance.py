import requests
from multiprocessing import Pool
from timeit import default_timer as timer

endpoint = 'http://localhost:8000'

response_time = []
task = {
    'prompt': "Once upon a time, ",
    'model': '/mnt/scratch/xiayao/cache/experiments/fmzip/compressed_models/3b0.5s/pythia-2.8b-deduped/task065_timetravel_consistent_sentence_classification/global_step48'
}

# task = {
#     'prompt': "Once upon a time, ",
#     'model': '/mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/pythia-2.8b-deduped/task065_timetravel_consistent_sentence_classification/global_step48'
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