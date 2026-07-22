"""Backend abstraction.

Both backends speak the OpenAI-compatible chat completions API, so vLLM and a
hosted provider sit behind the same interface. Swapping is a config change.
"""
from __future__ import annotations

from typing import Any, Protocol

import httpx

from .config import settings


class BackendError(RuntimeError):
    """Raised when a backend is unreachable, times out, or returns an error."""


class Backend(Protocol):
    name: str
    model: str

    async def chat(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        ...


class OpenAICompatibleBackend:
    """Works for vLLM's OpenAI-compatible server and for hosted providers."""

    def __init__(self, name: str, base_url: str, model: str, api_key: str = "") -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def chat(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise BackendError(f"{self.name} backend timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name} backend unreachable: {exc}") from exc

        if resp.status_code >= 400:
            raise BackendError(f"{self.name} backend returned {resp.status_code}")
        return resp.json()

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/models")
            return resp.status_code < 400
        except httpx.HTTPError:
            return False


def get_backend() -> OpenAICompatibleBackend:
    """Select the backend from configuration."""
    if settings.backend == "hosted":
        return OpenAICompatibleBackend(
            name="hosted",
            base_url=settings.hosted_url,
            model=settings.hosted_model,
            api_key=settings.hosted_api_key,
        )
    return OpenAICompatibleBackend(
        name="selfhosted",
        base_url=settings.selfhosted_url,
        model=settings.selfhosted_model,
    )
