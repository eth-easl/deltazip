import json
import time
import asyncio
import hashlib
from typing import Optional
from typing import Callable
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.routing import APIRoute
from timeit import default_timer as timer
from fastapi import FastAPI, Request, Response
from fmzip.rest.inference import InferenceService

app = FastAPI()
batch_size = 8
timeout = 3
task_queue = asyncio.Queue()

inference_model = InferenceService(
    provider='fmzip-mpm',
    base_model='EleutherAI/pythia-2.8b-deduped'
)

async def process_tasks():
    while True:
        batch = []
        try:
            # Try to collect batch_size number of tasks within timeout seconds
            for _ in range(batch_size):
                task = await asyncio.wait_for(task_queue.get(), timeout=timeout)
                batch.append(task)

        except asyncio.TimeoutError:
            # If timeout, get whatever tasks are available
            while not task_queue.empty():
                task = await task_queue.get()
                batch.append(task)

        if len(batch) > 0:
            print(f"Processing batch: {batch}")
            output = inference_model.generate(batch)
            for i, task in enumerate(batch):
                print(f"Executing task: {task}")
                results[task.id]['result'] = output[i]
                results[task.id]['event'].set()
            print("Batch processing completed.\n")

# Run the processing function as a background task
asyncio.create_task(process_tasks())

dhash = hashlib.md5()
results = {}

class InferenceTask(BaseModel):
    id: Optional[str] = ""
    prompt: str
    model: str
    response: Optional[str] = ""

@app.post("/inference", response_model=InferenceTask)
async def handle_request(inference_task: InferenceTask):
    dhash.update(json.dumps(inference_task.dict(), sort_keys=True).encode())
    inference_task.id = dhash.hexdigest()
    event = asyncio.Event()
    results[inference_task.id] = {'event': event, 'result': None}
    await task_queue.put(inference_task)
    await event.wait()
    response = results.pop(inference_task.id)['result']
    inference_task.response = response
    return inference_task