import json
model_size = 3 # billions of parameters
gpu_memory = 24 # GB
model_num = 100

# device required for dedicated GPU
dedicated_gpus = model_size * model_num * 2 / gpu_memory
print(dedicated_gpus)
# we only need 1
finicompress_gpus = 1

# tail latency for dedicated gpus
filename = 'artifact/results/poisson/3b/ar_0.75_4bits_64tokens.json'

# tail latency for dedicated approach
with open(filename, 'r') as fp:
    results = json.load(fp)

max_latency = 0
max_fini_latency = 0

result = [result for result in results if result['system']['name']=='hf'][0]
for res in result['results']:
    latency = res['response']['response']['measure']['tokenize_time'] + res['response']['response']['measure']['prepare_time'] + res['response']['response']['measure']['inference_time']
    print(latency)
    if latency > max_latency:
        max_latency = latency
print(f"Tail latency for dedicated approach: {max_latency:.3f}s")

# tail latency for finicompress
fc_result = [result for result in results if result['system']['name']=='fmzip' and result['system']['args']['batch_size']==4][0]

for res in fc_result['results']:
    print(res['time_elapsed'])
    if res['time_elapsed'] > max_fini_latency:
        max_fini_latency = res['time_elapsed']
print(f"Tail latency for FiniCompress: {max_fini_latency:.3f}s")