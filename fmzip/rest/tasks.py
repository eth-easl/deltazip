import asyncio
from dataclasses import dataclass, field
from typing import Any

class TaskQueue():
    def __init__(self, maxsize: int = 0):
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    def put_task(self, task: Any):
        self.queue.put(task)        
