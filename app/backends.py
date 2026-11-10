"""OpenAI-compatible backend with pooled connections and response validation."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Protocol

import httpx

from .config import Settings, settings


class BackendError(RuntimeError):
    """A backend is unreachable, timed out, or returned an HTTP error."""


class MalformedBackendResponse(BackendError):
    """A successful HTTP response does not contain supported text content."""


def response_content(raw: Any) -> str:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MalformedBackendResponse("Malformed backend response") from exc
    if not isinstance(content, str):
        raise MalformedBackendResponse("Malformed backend response")
    return content


class Backend(Protocol):
    name: str
    model: str
    base_url: str

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]: ...

    async def healthy(self) -> bool: ...

    async def aclose(self) -> None: ...


class OpenAICompatibleBackend:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        config: Settings | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.config = config or settings
        self._client: httpx.AsyncClient | None = None

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
        return self._client

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            async with asyncio.timeout(self.config.request_timeout_seconds):
                resp = await self.client().post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self.headers()
                )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise BackendError(f"{self.name} backend timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name} backend unreachable") from exc
        if resp.status_code >= 400:
            raise BackendError(f"{self.name} backend returned {resp.status_code}")
        try:
            raw = resp.json()
        except ValueError as exc:
            raise MalformedBackendResponse("Malformed backend response") from exc
        response_content(raw)
        return raw

    async def healthy(self) -> bool:
        try:
            async with asyncio.timeout(self.config.readiness_timeout_seconds):
                resp = await self.client().get(
                    f"{self.base_url}/models",
                    headers=self.headers(),
                    timeout=self.config.readiness_timeout_seconds,
                )
            if resp.status_code >= 400:
                return False
            data = resp.json()["data"]
            return isinstance(data, list) and any(
                isinstance(model, dict) and model.get("id") == self.model for model in data
            )
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
            return False

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


@lru_cache(maxsize=1)
def _backend_for_config(config: Settings) -> OpenAICompatibleBackend:
    if config.backend == "hosted":
        return OpenAICompatibleBackend(
            "hosted", config.hosted_url, config.hosted_model, config.hosted_api_key, config=config
        )
    return OpenAICompatibleBackend(
        "selfhosted", config.selfhosted_url, config.selfhosted_model, config=config
    )


def get_backend() -> OpenAICompatibleBackend:
    return _backend_for_config(settings)
