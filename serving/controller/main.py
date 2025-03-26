import httpx
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse
from .protocols import UpstreamRegistrationRequest
from .policies.core import uss, UpstreamServer

app = FastAPI()
DEFAULT_TIMEOUT = 12000
upstream_servers = []


@app.post("/proxy/{path:path}")
async def reverse_proxy(request: Request):
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000/", timeout=DEFAULT_TIMEOUT
    )
    # get model from request payload
    model_name = request.json().get("model")
    print("requested model:", model_name)
    path = request.url.path.replace("/proxy/", "/")
    url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
    req = client.build_request(
        request.method, url, headers=request.headers.raw, content=request.stream()
    )
    r = await client.send(req, stream=True)
    return StreamingResponse(
        r.aiter_raw(),
        status_code=r.status_code,
        headers=r.headers,
        background=BackgroundTask(r.aclose),
    )


@app.post("/register")
async def register_upstream(request: UpstreamRegistrationRequest):
    uss.append(UpstreamServer(url=request.ip_address, weight=request.weight))
