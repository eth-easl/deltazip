import os
import torch
import random
import asyncio
import threading
import subprocess
from queue import Queue
from loguru import logger
from typing import Optional
from fastapi import FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

from fmzip.rest.inference import InferenceService
from fmzip.rest.profile import profile_disk_io, get_gpu_name

app = FastAPI()
task_queue = Queue()

batch_size = int(os.environ.get("FMZIP_BATCH_SIZE", 1))
backend = os.environ.get("FMZIP_BACKEND", "hf")
base_model = os.environ.get("FMZIP_BASE_MODEL", "meta-llama/Llama-2-7b-hf")
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
num_gpus = len(cuda_visible_devices.split(","))

print(f"Available GPUs {num_gpus}")
inference_model = None
gpu_id_start = 1


def randomly_clear_disk_cache():
    # randomly clear disk cache with a probability of 0.5
    if random.random() < 1:
        subprocess.Popen("sudo echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)
        pass


def generation_thread(model, batch, gpu_id):
    logger.warning(f"processing {[x.id for x in batch]} on gpu {gpu_id}")
    output = model.generate(batch, gpu_id)

    for i, task in enumerate(batch):
        results[task.id]["result"] = output[i]
        results[task.id]["event"].set()
        # randomly_clear_disk_cache()


class BackgroundTasks(threading.Thread):
    async def _checking(self):
        while True:
            batch = []
            for _ in range(batch_size * num_gpus):
                try:
                    task = task_queue.get_nowait()
                    batch.append(task)
                except:
                    break
            if len(batch) > 0:
                # sort by id
                batch = sorted(batch, key=lambda x: int(x.id))
                if num_gpus == 1:
                    output = inference_model.generate(batch, "0")
                    for i, task in enumerate(batch):
                        results[task.id]["result"] = output[i]
                        results[task.id]["event"].set()
                        randomly_clear_disk_cache()
                else:
                    # split batch into sub-batches, per gpu, evenly
                    sub_batches = [[] for _ in range(num_gpus)]
                    for task in batch:
                        if inference_model.find_model(task.model) is not None:
                            sub_batches[inference_model.find_model(task.model)].append(
                                task
                            )
                        else:
                            # if model is not found, send the gpu with minimal tasks
                            minimal_queue = sorted(sub_batches, key=lambda x: len(x))
                            minimal_queue[gpu_id_start].append(task)
                    # run inference on each gpu, as a new thread
                    # in case we know the model being requested is already on certain GPU, we can directly send the batch to that GPUs
                    threads = []
                    print(sub_batches)
                    for idx, sb in enumerate(sub_batches):
                        if len(sb) > 0:
                            thread = threading.Thread(
                                target=generation_thread,
                                args=(
                                    inference_model,
                                    sb,
                                    idx,
                                ),
                            )
                            threads.append(thread)
                            thread.start()
                    [thread.join() for thread in threads]

    def run(self, *args, **kwargs):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._checking())


results = {}


class InferenceTask(BaseModel):
    id: Optional[int] = ""
    prompt: str
    model: str
    response: Optional[dict] = {}
    timestamp: float


class RestartRequest(BaseModel):
    base_model: str
    backend: str
    backend_args: dict
    mapping: dict
    gen_configs: dict


@app.post("/inference", response_model=InferenceTask)
async def handle_request(inference_task: InferenceTask):
    event = asyncio.Event()
    results[inference_task.id] = {"event": event, "result": None}
    task_queue.put(inference_task)
    await event.wait()
    response = results.pop(inference_task.id)["result"]
    inference_task.response = response
    return inference_task


@app.post("/restart")
async def handle_restart(restart_request: RestartRequest):
    logger.info(
        f"Server is reconfigured to use {restart_request.backend} backend with {restart_request.base_model} base model"
    )
    logger.info(f"Restart args: {restart_request}")
    global inference_model
    global batch_size
    del inference_model
    torch.cuda.empty_cache()
    inference_model = InferenceService(
        base_model=restart_request.base_model,
        backend=restart_request.backend,
        backend_args=restart_request.backend_args,
        mapping=restart_request.mapping,
        gen_configs=restart_request.gen_configs,
    )
    batch_size = inference_model.batch_size
    return {"status": "success"}


@app.get("/status")
async def handle_status():
    return {"disk_bandwidth": profile_disk_io(), "gpu_name": get_gpu_name()}


@app.on_event("startup")
async def startup_event():
    t = BackgroundTasks()
    t.start()
