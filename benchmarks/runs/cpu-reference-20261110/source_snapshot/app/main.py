"""FastAPI gateway: routing, caching, liveness, and backend readiness."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .backends import BackendError, MalformedBackendResponse, get_backend, response_content
from .cache import cache, cache_key, valid_cache_value
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await cache.aclose()
        await get_backend().aclose()


app = FastAPI(title="LLM Serving Gateway", version=settings.app_version, lifespan=lifespan)


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    use_cache: bool = True


class ChatResponse(BaseModel):
    content: str
    backend: str
    model: str
    cached: bool
    latency_ms: float
    app_version: str


@app.get("/health")
async def health() -> dict:
    """Process liveness; deliberately makes no external dependency calls."""
    backend = get_backend()
    return {
        "status": "ok",
        "backend": backend.name,
        "model": backend.model,
        "version": settings.app_version,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    backend = get_backend()
    reachable = await backend.healthy()
    return JSONResponse(
        status_code=200 if reachable else 503,
        content={
            "status": "ready" if reachable else "unavailable",
            "backend": backend.name,
            "model": backend.model,
            "backend_reachable": reachable,
            "version": settings.app_version,
        },
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    cache_namespace: str = Header(default="", alias="X-Cache-Namespace", max_length=128),
) -> ChatResponse:
    started = time.perf_counter()
    backend = get_backend()
    messages = [m.model_dump() for m in req.messages]
    key = cache_key(
        backend.model,
        messages,
        req.max_tokens,
        backend_name=backend.name,
        base_url=backend.base_url,
        model_revision=settings.model_revision,
        inference_dtype=settings.inference_dtype,
        temperature=req.temperature,
        seed=req.seed,
        namespace=cache_namespace,
    )
    if req.use_cache:
        hit = await cache.get(key)
        if valid_cache_value(hit):
            return ChatResponse(
                content=hit["content"],
                backend=backend.name,
                model=backend.model,
                cached=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                app_version=settings.app_version,
            )
    try:
        raw = await backend.chat(messages, req.max_tokens, req.temperature, req.seed)
        content = response_content(raw)
    except MalformedBackendResponse as exc:
        raise HTTPException(status_code=502, detail="Malformed backend response") from exc
    except BackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if req.use_cache:
        await cache.set(key, {"content": content})
    return ChatResponse(
        content=content,
        backend=backend.name,
        model=backend.model,
        cached=False,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        app_version=settings.app_version,
    )
