"""FastAPI gateway: routing, caching, health, and failure handling."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .backends import BackendError, get_backend
from .cache import cache, cache_key
from .config import settings

app = FastAPI(title="LLM Serving Gateway", version=settings.app_version)


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
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
    backend = get_backend()
    return {
        "status": "ok",
        "backend": backend.name,
        "model": backend.model,
        "backend_reachable": await backend.healthy(),
        "version": settings.app_version,
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    backend = get_backend()
    messages = [m.model_dump() for m in req.messages]
    key = cache_key(backend.model, messages, req.max_tokens)

    if req.use_cache:
        hit = await cache.get(key)
        if hit is not None:
            return ChatResponse(
                content=hit["content"],
                backend=backend.name,
                model=backend.model,
                cached=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                app_version=settings.app_version,
            )

    try:
        raw = await backend.chat(messages, req.max_tokens)
    except BackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Malformed backend response") from exc

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
