"""Dependency failures and cache isolation, without Redis/GPU/network services."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.backends import BackendError, MalformedBackendResponse, OpenAICompatibleBackend
from app.cache import ResponseCache, cache_key
from app.config import Settings

MESSAGES = [{"role": "user", "content": "a fixed prompt"}]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def key(**overrides):
    parameters = {
        "model": "same-model",
        "messages": MESSAGES,
        "max_tokens": 32,
        "backend_name": "selfhosted",
        "base_url": "http://a/v1",
        "model_revision": "weights-v1",
        "inference_dtype": "float32",
        "temperature": 0.0,
        "seed": 17,
        "namespace": "run-1",
    }
    parameters.update(overrides)
    return cache_key(**parameters)


@pytest.mark.parametrize(
    "different",
    [
        {"backend_name": "hosted"},
        {"base_url": "http://another-server/v1"},
        {"model_revision": "weights-v2"},
        {"inference_dtype": "float16"},
        {"temperature": 0.5},
        {"seed": 18},
        {"namespace": "run-2"},
    ],
)
def test_cache_key_does_not_share_across_response_affecting_context(different):
    assert key() != key(**different)
    assert key(base_url="http://a/v1/") == key()


class RedisStub:
    def __init__(self, raw=None, *, hang=False, fail=False):
        self.raw = raw
        self.hang = hang
        self.fail = fail
        self.writes = []

    async def _wait(self):
        if self.fail:
            raise ConnectionError("Redis unavailable")
        if self.hang:
            await asyncio.Event().wait()

    async def get(self, name):
        await self._wait()
        return self.raw

    async def setex(self, *args):
        await self._wait()
        self.writes.append(args)


@pytest.mark.anyio
@pytest.mark.parametrize("raw", ["{broken", "[]", '"string"', '{"content":null}', "{}"])
async def test_corrupt_redis_entry_is_a_cache_miss(raw):
    cache = ResponseCache()
    cache._client = RedisStub(raw)
    assert await cache.get("key") is None


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["fail", "hang"])
async def test_redis_outage_is_a_bounded_noop(monkeypatch, mode):
    monkeypatch.setattr("app.cache.settings", Settings(cache_io_timeout_seconds=0.02))
    cache = ResponseCache()
    cache._client = RedisStub(**{mode: True})
    # Outer timeout catches a missing implementation deadline; no timing percentile claims.
    async with asyncio.timeout(0.5):
        assert await cache.get("key") is None
        assert await cache.set("key", {"content": "valid"}) is None
    assert cache._client.writes == []


@pytest.mark.anyio
async def test_invalid_value_never_written_to_redis():
    cache = ResponseCache()
    cache._client = RedisStub()
    await cache.set("key", {"content": None})
    assert cache._client.writes == []


def make_backend(handler, **config):
    backend = OpenAICompatibleBackend(
        "hosted", "http://backend/v1", "expected-model", "secret", config=Settings(**config)
    )
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return backend


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"null",
        b"[]",
        b'{"choices":[]}',
        b'{"choices":[{"message":{"content":null}}]}',
        b'{"choices":[{"message":{"content":["unsupported"]}}]}',
    ],
)
async def test_successful_http_with_malformed_content_is_a_backend_contract_error(body):
    backend = make_backend(lambda request: httpx.Response(200, content=body))
    try:
        with pytest.raises(MalformedBackendResponse):
            await backend.chat(MESSAGES, 32)
    finally:
        await backend.aclose()


@pytest.mark.anyio
async def test_backend_forwards_generation_controls_and_reuses_client():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "text"}}]})

    backend = make_backend(handler)
    original_client = backend.client()
    try:
        await backend.chat(MESSAGES, 32, temperature=0.3, seed=42)
        await backend.chat(MESSAGES, 32)
        assert backend.client() is original_client
        first, second = [json.loads(request.content) for request in seen]
        assert first["temperature"] == 0.3 and first["seed"] == 42
        assert second["temperature"] == 0.0 and "seed" not in second
        assert seen[0].headers["authorization"] == "Bearer secret"
    finally:
        await backend.aclose()
    assert original_client.is_closed


@pytest.mark.anyio
async def test_total_backend_deadline_bounds_a_nonresponding_upstream():
    async def handler(request):
        await asyncio.Event().wait()

    backend = make_backend(handler, request_timeout_seconds=0.02)
    try:
        async with asyncio.timeout(0.5):
            with pytest.raises(BackendError, match="timed out"):
                await backend.chat(MESSAGES, 32)
    finally:
        await backend.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("advertised_model, expected", [("expected-model", True), ("other", False)])
async def test_readiness_is_authenticated_and_checks_configured_model(advertised_model, expected):
    def handler(request):
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json={"data": [{"id": advertised_model}]})

    backend = make_backend(handler)
    try:
        assert await backend.healthy() is expected
    finally:
        await backend.aclose()
