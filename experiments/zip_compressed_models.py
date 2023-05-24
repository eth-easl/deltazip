import os
from multiprocessing import Pool

model_size = 'opt-1.3b'

os.makedirs(f'.cache/zipped_models/{model_size}', exist_ok=True)

compressed_models = os.listdir(f'.cache/compressed_models/{model_size}')

def compress_each(cm):
    input_file = os.path.join(f'.cache/compressed_models/{model_size}', cm)
    print("Compressing", input_file)
    otuput_file = os.path.join(f'.cache/zipped_models/{model_size}', cm+".7z")
    os.system(f"7z a -mx=9 {otuput_file} {input_file}/*")

with Pool(16) as p:
    p.map(compress_each, compressed_models)