"""Configuration. Backend selection is a config change, not a code change."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend: str = os.getenv("LLM_BACKEND", "selfhosted")  # "selfhosted" | "hosted"
    selfhosted_url: str = os.getenv("VLLM_URL", "http://localhost:8001/v1")
    selfhosted_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    hosted_url: str = os.getenv("HOSTED_URL", "https://api.openai.com/v1")
    hosted_model: str = os.getenv("HOSTED_MODEL", "gpt-4o-mini")
    hosted_api_key: str = os.getenv("HOSTED_API_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "3600"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT", "60"))
    app_version: str = os.getenv("APP_VERSION", "0.1.0")


settings = Settings()
