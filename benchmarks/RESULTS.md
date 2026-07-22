# Benchmark Results

> Every number below is measured. Nothing is estimated. Fill each `TBD` after an
> actual run, and delete any row that was not run.

## Environment

| Field | Value |
| --- | --- |
| GPU | TBD (e.g. NVIDIA A10G, 24 GB) |
| GPU memory | TBD |
| Host | TBD (e.g. RunPod A10G, 8 vCPU, 32 GB RAM) |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| dtype | TBD (e.g. bfloat16) |
| vLLM version | TBD |
| Gateway | FastAPI + uvicorn, 1 worker |
| Redis | TBD version, single node, local |
| Date | TBD |

## A. Model serving (vLLM, gateway bypassed)

Command:

```bash
vllm bench serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --base-url http://localhost:8001 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 256 --random-output-len 128 \
  --max-concurrency 16
```

| Metric | Value |
| --- | --- |
| Request throughput (req/s) | TBD |
| Output token throughput (tok/s) | TBD |
| Median TTFT (ms) | TBD |
| P99 TTFT (ms) | TBD |
| Median TPOT (ms) | TBD |
| P99 TPOT (ms) | TBD |
| Median end-to-end latency (ms) | TBD |
| Successful requests | TBD / TBD |

Repeat at concurrency 1, 8, 16, 32 to show where saturation begins.

## B. Full gateway path (Locust)

```bash
locust -f benchmarks/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 3m --headless --csv=benchmarks/run
```

| Concurrent users | Path | Req/s | P50 (ms) | P95 (ms) | Error rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | warm cache | TBD | TBD | TBD | TBD |
| 1 | cold cache | TBD | TBD | TBD | TBD |
| 10 | warm cache | TBD | TBD | TBD | TBD |
| 10 | cold cache | TBD | TBD | TBD | TBD |
| 50 | warm cache | TBD | TBD | TBD | TBD |
| 50 | cold cache | TBD | TBD | TBD | TBD |

## C. Derived findings

| Question | Answer |
| --- | --- |
| Cache hit latency reduction vs. cold | TBD % |
| Gateway overhead (no-cache path minus direct vLLM P50) | TBD ms |
| Concurrency at which P95 exceeds 2x the single-user P50 | TBD |
| Behavior when the model server is stopped | TBD (expect 503, no hang) |

## Limitations

- Single-node, single-GPU. No multi-replica or multi-node results.
- Synthetic prompt distribution; real traffic would differ in length and cache hit rate.
- Load generator ran on the same host, so network latency is not represented.
- No sustained soak test; results are short-duration.
