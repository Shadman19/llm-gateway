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
    assert "backend_reachable" not in body


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
    assert cache_key("m", msgs, 32, backend_name="a", base_url="http://a/v1") == cache_key(
        "m", msgs, 32, backend_name="a", base_url="http://a/v1"
    )
    assert cache_key("m", msgs, 32, backend_name="a", base_url="http://a/v1") != cache_key(
        "m", msgs, 64, backend_name="a", base_url="http://a/v1"
    )
    assert cache_key("m", msgs, 32, backend_name="a", base_url="http://a/v1") != cache_key(
        "other", msgs, 32, backend_name="a", base_url="http://a/v1"
    )


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
    async def bad_chat(messages, max_tokens, temperature=0.0, seed=None):
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


def test_liveness_does_not_depend_on_backend_but_readiness_does(client, fake_backend, monkeypatch):
    async def unavailable():
        return False

    monkeypatch.setattr(fake_backend, "healthy", unavailable)
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["backend_reachable"] is False


@pytest.mark.parametrize("content", [None, [], {}, 42])
def test_nonstring_backend_content_is_502_and_never_cached(
    client, fake_backend, memory_cache, content
):
    fake_backend.response_text = content
    assert client.post("/v1/chat", json=BODY).status_code == 502
    assert memory_cache.store == {}


@pytest.mark.parametrize("different", [{"temperature": 0.5}, {"seed": 1}])
def test_generation_parameters_are_not_served_from_other_parameters_cache(
    client, fake_backend, different
):
    client.post("/v1/chat", json=BODY)
    response = client.post("/v1/chat", json=dict(BODY, **different))
    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert fake_backend.calls == 2


def test_cache_namespace_is_separate_and_has_a_size_limit(client, fake_backend):
    first = client.post("/v1/chat", json=BODY, headers={"X-Cache-Namespace": "run-1"})
    second = client.post("/v1/chat", json=BODY, headers={"X-Cache-Namespace": "run-2"})
    repeated = client.post("/v1/chat", json=BODY, headers={"X-Cache-Namespace": "run-1"})
    assert first.json()["cached"] is False and second.json()["cached"] is False
    assert repeated.json()["cached"] is True and fake_backend.calls == 2
    assert (
        client.post("/v1/chat", json=BODY, headers={"X-Cache-Namespace": "x" * 129}).status_code
        == 422
    )


def test_same_model_at_another_endpoint_cannot_reuse_cached_response(client, fake_backend):
    client.post("/v1/chat", json=BODY)
    fake_backend.base_url = "http://different-deployment/v1"
    fake_backend.response_text = "different deployment"
    response = client.post("/v1/chat", json=BODY).json()
    assert response["cached"] is False
    assert response["content"] == "different deployment"
    assert fake_backend.calls == 2
