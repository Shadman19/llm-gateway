"""Redis response cache. Degrades to a no-op if Redis is unavailable."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as redis

from .config import settings


def cache_key(model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens}, sort_keys=True
    )
    return "llmcache:" + hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._client: redis.Redis | None = None

    async def client(self) -> redis.Redis | None:
        if self._client is None:
            try:
                self._client = redis.from_url(self._url, decode_responses=True)
                await self._client.ping()
            except Exception:
                self._client = None
        return self._client

    async def get(self, key: str) -> dict[str, Any] | None:
        client = await self.client()
        if client is None:
            return None
        try:
            raw = await client.get(key)
        except Exception:
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        client = await self.client()
        if client is None:
            return
        try:
            await client.setex(key, settings.cache_ttl_seconds, json.dumps(value))
        except Exception:
            return


cache = ResponseCache()
