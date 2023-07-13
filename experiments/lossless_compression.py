import os
import pandas as pd
from loguru import logger
from src.lossless.service import CompressedInferenceService
from argparse import ArgumentParser

if __name__=="__main__":
    base_model = "facebook/opt-6.7b"
    service = CompressedInferenceService(
        base_model=base_model,
        dtype='fp16'
    )
    target_model = os.listdir("/home/xzyao/Documents/cache/ni_models/opt-6.7b")
    target_models = [os.path.join("/home/xzyao/Documents/cache/ni_models/opt-6.7b", i) for i in target_model]
    results = []
    for target_model in target_models:
        dest = os.path.join(".cache", "compressed_model", target_model.replace("/", "-"))
        os.makedirs(dest, exist_ok=True)
        compression_rate, time_spent = service.compress_model(
            target_model=target_model,
            dest=dest,
            low_gpu_mem=True,
            delta=True
        )
        logger.info("Compression rate: {}x".format(compression_rate))
        logger.info("Time spent: {}s".format(time_spent))
        results.append({
            'target_model': target_model,
            'compression_rate': compression_rate,
            'time_spent': time_spent,
            'dtype': 'fp16',
            'model_size': '6.7'
        })
    pd.DataFrame(results).to_csv(".cache/results/opt-6.7b.csv")