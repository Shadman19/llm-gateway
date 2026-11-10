"""Best-effort Redis cache with isolated keys and bounded dependency I/O."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

import redis.asyncio as redis
from redis.exceptions import RedisError

from .config import settings

logger = logging.getLogger(__name__)


def cache_key(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    backend_name: str,
    base_url: str,
    model_revision: str = "unspecified",
    inference_dtype: str = "unspecified",
    temperature: float = 0.0,
    seed: int | None = None,
    namespace: str = "",
) -> str:
    payload = json.dumps(
        {
            "schema": 2,
            "backend": backend_name,
            "base_url": base_url.rstrip("/"),
            "model": model,
            "model_revision": model_revision,
            "inference_dtype": inference_dtype,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "seed": seed,
            "namespace": namespace,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "llmcache:v2:" + hashlib.sha256(payload.encode()).hexdigest()


def valid_cache_value(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("content"), str)


class ResponseCache:
    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._client: redis.Redis | None = None

    async def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=settings.cache_io_timeout_seconds,
                socket_timeout=settings.cache_io_timeout_seconds,
            )
        return self._client

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            async with asyncio.timeout(settings.cache_io_timeout_seconds):
                client = await self.client()
                raw = await client.get(key)
            value = json.loads(raw) if raw else None
        except (RedisError, OSError, ValueError, TypeError):
            return None
        return value if valid_cache_value(value) else None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        if not valid_cache_value(value):
            return
        try:
            encoded = json.dumps(value)
            async with asyncio.timeout(settings.cache_io_timeout_seconds):
                client = await self.client()
                await client.setex(key, settings.cache_ttl_seconds, encoded)
        except (RedisError, OSError, ValueError, TypeError):
            return

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                async with asyncio.timeout(settings.cache_io_timeout_seconds):
                    await client.aclose()
            except (RedisError, OSError, ValueError, TypeError):
                logger.warning("Redis cache shutdown failed")


cache = ResponseCache()
