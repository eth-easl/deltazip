import os
from fmzip.lossless.service import CompressedInferenceService
from loguru import logger

base_model = "EleutherAI/gpt-j-6b"
target_model = "nlpulse/gpt-j-6b-english_quotes"
dest = os.path.join(".cache", "compressed_model", target_model.replace("/", "-"))
os.makedirs(dest, exist_ok=True)

service = CompressedInferenceService(base_model=base_model, dtype="fp16")

compression_rate, time_spent = service.compress_model(
    target_model=target_model, dest=dest, low_gpu_mem=True, delta=True
)

logger.info("Compression rate: {}x".format(compression_rate))
logger.info("Time spent: {}s".format(time_spent))
