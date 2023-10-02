import os
import json
import torch
import asyncio
import hashlib
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fmzip.rest.inference import InferenceService
from loguru import logger
import threading
from queue import Queue

app = FastAPI()
task_queue = Queue()
is_busy = False

batch_size = int(os.environ.get('FMZIP_BATCH_SIZE', 2))
backend = os.environ.get('FMZIP_BACKEND', 'hf')
base_model = os.environ.get("FMZIP_BASE_MODEL", "meta-llama/Llama-2-7b-hf")

inference_model = None

class BackgroundTasks(threading.Thread):
    async def _checking(self):
        while True:
            batch = []
            # Try to collect batch_size number of tasks within timeout seconds
            for _ in range(batch_size):
                try:
                    task = task_queue.get_nowait()
                    batch.append(task)
                except:
                    break
            
            if len(batch) > 0:
                if inference_model.provider=='hf':
                    for query in batch:
                        output = inference_model.generate([query])
                        for i, task in enumerate([query]):
                            results[task.id]['result'] = output[i]
                            results[task.id]['event'].set()
                else:
                    output = inference_model.generate(batch)
                    for i, task in enumerate(batch):
                        results[task.id]['result'] = output[i]
                        results[task.id]['event'].set()

    def run(self,*args,**kwargs):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._checking())


dhash = hashlib.md5()
results = {}

class InferenceTask(BaseModel):
    id: Optional[str] = ""
    prompt: str
    model: str
    response: Optional[str] = ""

class RestartRequest(BaseModel):
    backend: str
    base_model: str
    batch_size: int = 2

@app.post("/inference", response_model=InferenceTask)
async def handle_request(inference_task: InferenceTask):
    # dhash.update(json.dumps(inference_task.model_dump(), sort_keys=True).encode())
    # inference_task.id = dhash.hexdigest()
    event = asyncio.Event()
    results[inference_task.id] = {'event': event, 'result': None}
    task_queue.put(inference_task)
    await event.wait()
    response = results.pop(inference_task.id)['result']
    inference_task.response = response
    return inference_task

@app.post("/restart")
async def handle_restart(restart_request: RestartRequest):
    logger.info(f"Server is reconfigured to use {restart_request.backend} backend with {restart_request.base_model} base model")
    global inference_model
    global batch_size
    del inference_model
    torch.cuda.empty_cache()
    inference_model = InferenceService(
        provider = restart_request.backend,
        base_model = restart_request.base_model,
        batch_size = restart_request.batch_size,
    )
    batch_size = restart_request.batch_size
    
    return {"status": "success"}

@app.on_event("startup")
async def startup_event():
    t = BackgroundTasks()
    t.start()