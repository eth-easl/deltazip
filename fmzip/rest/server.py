import time
import asyncio
from typing import Callable
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.routing import APIRoute
from timeit import default_timer as timer
from fastapi import FastAPI, Request, Response

class LockableRoute(APIRoute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = asyncio.Lock()

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            start = timer()
            await self.lock.acquire()
            response: Response = await original_route_handler(request)
            self.lock.release()
            end = timer()
            response.headers["X-Response-Time"] = str(end-start)
            return response

        return custom_route_handler



app = FastAPI()
app.router.route_class = LockableRoute

@app.get("/inference")
async def handle_request():
    time.sleep(0.5)
    return {"message": 'done'}
