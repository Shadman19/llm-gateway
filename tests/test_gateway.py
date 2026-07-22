"""Gateway tests: routing, caching, validation, failure handling."""

from __future__ import annotations

import pytest

from app.backends import BackendError, OpenAICompatibleBackend, get_backend
from app.cache import cache_key
from app.config import Settings

BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}


# --- health -----------------------------------------------------------------


def test_health_reports_backend(client, fake_backend):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == fake_backend.name
    assert body["backend_reachable"] is True


# --- routing ----------------------------------------------------------------


def test_chat_returns_backend_content(client, fake_backend):
    fake_backend.response_text = "routed"
    resp = client.post("/v1/chat", json=BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "routed"
    assert body["cached"] is False
    assert body["model"] == fake_backend.model


def test_backend_selection_is_config_driven(monkeypatch):
    monkeypatch.setattr("app.backends.settings", Settings(backend="hosted"))
    assert get_backend().name == "hosted"
    monkeypatch.setattr("app.backends.settings", Settings(backend="selfhosted"))
    assert get_backend().name == "selfhosted"


# --- caching ----------------------------------------------------------------


def test_cache_miss_then_hit(client, fake_backend):
    first = client.post("/v1/chat", json=BODY).json()
    second = client.post("/v1/chat", json=BODY).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert fake_backend.calls == 1, "cache hit must not reach the backend"


def test_cache_can_be_bypassed(client, fake_backend):
    client.post("/v1/chat", json=BODY)
    body = dict(BODY, use_cache=False)
    resp = client.post("/v1/chat", json=body).json()
    assert resp["cached"] is False
    assert fake_backend.calls == 2


def test_cache_key_is_stable_and_distinguishing():
    msgs = [{"role": "user", "content": "hello"}]
    assert cache_key("m", msgs, 32) == cache_key("m", msgs, 32)
    assert cache_key("m", msgs, 32) != cache_key("m", msgs, 64)
    assert cache_key("m", msgs, 32) != cache_key("other", msgs, 32)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "bogus", "content": "x"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 999999},
    ],
)
def test_invalid_requests_rejected(client, payload):
    assert client.post("/v1/chat", json=payload).status_code == 422


# --- failure handling -------------------------------------------------------


def test_backend_failure_returns_503(client, fake_backend):
    fake_backend.raise_error = BackendError("backend down")
    resp = client.post("/v1/chat", json=BODY)
    assert resp.status_code == 503


def test_malformed_backend_response_returns_502(client, fake_backend, monkeypatch):
    async def bad_chat(messages, max_tokens):
        return {"unexpected": True}

    monkeypatch.setattr(fake_backend, "chat", bad_chat)
    assert client.post("/v1/chat", json=BODY).status_code == 502


@pytest.mark.anyio
async def test_backend_timeout_raises_backend_error(monkeypatch):
    import httpx

    class TimeoutClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: TimeoutClient())
    backend = OpenAICompatibleBackend("selfhosted", "http://x/v1", "m")
    with pytest.raises(BackendError):
        await backend.chat([{"role": "user", "content": "x"}], 8)


@pytest.fixture
def anyio_backend():
    return "asyncio"
