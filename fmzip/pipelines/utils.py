from pynvml import nvmlInit, nvmlDeviceGetCount
from loguru import logger

nvml_is_initialized = False


def initialize():
    global nvml_is_initialized
    if not nvml_is_initialized:
        nvmlInit()
        nvml_is_initialized = True
    else:
        logger.info("nvml is already initialized")


def get_gpu_count():
    initialize()
    return nvmlDeviceGetCount()
