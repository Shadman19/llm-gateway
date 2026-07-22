# LLM Serving Gateway

![CI](https://github.com/Shadman19/llm-gateway/actions/workflows/ci.yml/badge.svg)

A benchmarked LLM serving gateway that routes requests to either a self-hosted
open-weights model (Qwen2.5-1.5B-Instruct via vLLM) or a hosted API, with Redis
response caching, request validation, health checks, and failure handling.
Backend selection is a configuration change, not a code change.

Production-oriented and benchmarked. Not production-grade: it has never carried
real user traffic and has no uptime, security review, or operational ownership
behind it.

## Measured results

Full methodology and hardware in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Metric | Value |
| --- | --- |
| Median TTFT | TBD ms |
| Output token throughput | TBD tok/s |
| Gateway P95, cold cache, 50 users | TBD ms |
| Gateway P95, warm cache, 50 users | TBD ms |
| Cache latency reduction | TBD % |
| Error rate at peak concurrency | TBD % |
| Hardware | TBD |

## Quick start

```bash
# 1. Serve the model (GPU host)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8001

# 2. Start gateway and Redis
docker compose up --build

# 3. Call it
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain KV caching briefly."}],"max_tokens":128}'
```

Response:

```json
{
  "content": "...",
  "backend": "selfhosted",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "cached": false,
  "latency_ms": 0.0,
  "app_version": "0.1.0"
}
```

Switch to a hosted backend with no code change:

```bash
LLM_BACKEND=hosted HOSTED_API_KEY=sk-... docker compose up
```

## Architecture

```
client -> FastAPI gateway -> Redis cache (hit -> return)
                          -> backend selector (config)
                               -> vLLM OpenAI-compatible server (self-hosted)
                               -> hosted provider API
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage spans routing, config-driven backend selection, cache hit and miss,
cache bypass, cache key stability, request validation, backend failure (503),
malformed backend responses (502), and backend timeout handling.

CI mocks the backend so it runs without a GPU. The real GPU smoke test is run
manually and recorded in `benchmarks/RESULTS.md`.

## Design decisions

- **OpenAI-compatible interface for both backends.** vLLM exposes this natively,
  so one client path serves both and swapping is configuration only.
- **Cache degrades to a no-op.** If Redis is unreachable the gateway serves
  uncached rather than failing, so a cache outage is not an availability outage.
- **Warm and cold cache measured separately.** Blending them into a single mean
  latency would make the cache look like a model speedup, which it is not.
- **503 rather than a hang on backend failure.** Timeouts are bounded and surfaced.

## Limitations

- Single GPU, single gateway replica. No multi-node results.
- No authentication or rate limiting; both would be required before real exposure.
- No streaming responses yet.
- Benchmarks are short-duration and synthetic.

## Background

Built to extend earlier work on LLM serving infrastructure with real self-hosted
inference and reproducible performance measurement.
