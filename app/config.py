"""Immutable process configuration for serving and bounded dependency I/O."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend: str = os.getenv("LLM_BACKEND", "selfhosted")
    selfhosted_url: str = os.getenv("VLLM_URL", "http://localhost:8001/v1")
    selfhosted_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    hosted_url: str = os.getenv("HOSTED_URL", "https://api.openai.com/v1")
    hosted_model: str = os.getenv("HOSTED_MODEL", "gpt-4o-mini")
    hosted_api_key: str = os.getenv("HOSTED_API_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "3600"))
    cache_io_timeout_seconds: float = float(os.getenv("CACHE_TIMEOUT", "0.25"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT", "60"))
    readiness_timeout_seconds: float = float(os.getenv("READINESS_TIMEOUT", "5"))
    model_revision: str = os.getenv("MODEL_REVISION", "unspecified")
    inference_dtype: str = os.getenv("INFERENCE_DTYPE", "unspecified")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")

    def __post_init__(self) -> None:
        if self.backend not in {"selfhosted", "hosted"}:
            raise ValueError("LLM_BACKEND must be selfhosted or hosted")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("CACHE_TTL must be positive")
        for value in (
            self.cache_io_timeout_seconds,
            self.request_timeout_seconds,
            self.readiness_timeout_seconds,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Dependency timeouts must be finite and positive")


settings = Settings()
