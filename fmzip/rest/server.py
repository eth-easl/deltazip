import os
import json
import asyncio
import hashlib
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fmzip.rest.inference import InferenceService

app = FastAPI()
task_queue = asyncio.Queue()
is_busy = False

batch_size = int(os.environ.get('FMZIP_BATCH_SIZE', 2))
backend = os.environ.get('FMZIP_BACKEND', 'hf')
base_model = os.environ.get("FMZIP_BASE_MODEL", "meta-llama/Llama-2-7b-hf")

inference_model = InferenceService(
    provider=backend,
    base_model=base_model
)

async def process_tasks():
    global is_busy
    while True:
        batch = []
        # Try to collect batch_size number of tasks within timeout seconds
        if not is_busy:
            for _ in range(batch_size):
                task = await asyncio.wait_for(task_queue.get(), timeout=100)
                batch.append(task)
            is_busy = True
        if len(batch) > 0:
            print(f"Processing batch: {batch}")
            output = inference_model.generate(batch)
            for i, task in enumerate(batch):
                results[task.id]['result'] = output[i]
                results[task.id]['event'].set()
            print("Batch processing completed.\n")
            is_busy = False
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