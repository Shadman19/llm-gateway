"""Fixtures. The backend is mocked so CI runs without a GPU."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeBackend:
    """Stands in for vLLM. Records calls so tests can assert on routing."""

    def __init__(self, name="selfhosted", model="Qwen/Qwen2.5-1.5B-Instruct"):
        self.name = name
        self.model = model
        self.calls = 0
        self.raise_error: Exception | None = None
        self.response_text = "fake response"

    async def chat(self, messages, max_tokens):
        self.calls += 1
        if self.raise_error:
            raise self.raise_error
        return {"choices": [{"message": {"content": self.response_text}}]}

    async def healthy(self):
        return True


class MemoryCache:
    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value):
        self.store[key] = value


@pytest.fixture
def fake_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr("app.main.get_backend", lambda: backend)
    return backend


@pytest.fixture
def memory_cache(monkeypatch):
    c = MemoryCache()
    monkeypatch.setattr("app.main.cache", c)
    return c


@pytest.fixture
def client(fake_backend, memory_cache):
    return TestClient(app)
