"""Matched-workload serving measurements; synthetic harness checks are not LLM results.

Run separate processes/runs for direct-json, gateway-bypass, gateway-cold and
gateway-warm. The gateway is non-streaming: its TTFT is deliberately unknown.
Use direct-stream only for a separately labelled SSE/TTFT experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

MODES = (
    "direct-json",
    "direct-stream",
    "gateway-bypass",
    "gateway-cold",
    "gateway-warm",
)


@dataclass(frozen=True)
class Prompt:
    id: str
    messages: list[dict[str, str]]
    max_tokens: int = 128
    expected: str | None = None


@dataclass(frozen=True)
class Config:
    url: str
    mode: str
    run_id: str
    model: str = ""
    requests: int = 200
    concurrency: int = 1
    timeout_seconds: float = 60.0
    retries: int = 0
    cache_namespace: str = ""
    temperature: float = 0.0
    seed: int | None = 0
    arrival_rate: float | None = None

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError("unknown mode")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", self.run_id):
            raise ValueError("run_id must be 1-48 simple identifier characters")
        if not 1 <= self.requests <= 100000 or not 1 <= self.concurrency <= 1000:
            raise ValueError("requests/concurrency outside bounded limits")
        if (
            not 0 <= self.retries <= 3
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("retries must be 0-3 and timeout positive")
        if (
            not 0 <= self.temperature <= 2
            or self.seed is not None
            and not 0 <= self.seed <= 2147483647
        ):
            raise ValueError("invalid generation settings")
        if self.arrival_rate is not None and (
            not math.isfinite(self.arrival_rate) or self.arrival_rate <= 0
        ):
            raise ValueError("arrival_rate must be finite and positive")
        if self.mode.startswith("direct") and not self.model:
            raise ValueError("direct measurements require an explicit model")
        if self.mode in ("gateway-cold", "gateway-warm"):
            if not self.cache_namespace:
                raise ValueError(
                    "cache measurements require an isolated cache_namespace"
                )
            if len(self.namespace()) > 128:
                raise ValueError(
                    "combined cache namespace exceeds gateway 128-character limit"
                )
        if self.mode == "gateway-warm" and self.temperature != 0:
            raise ValueError(
                "warm-cache replay uses temperature=0; stochastic replay is a different experiment"
            )

    def namespace(self) -> str:
        return f"{self.cache_namespace}/{self.run_id}/{self.mode}"


def load_prompts(path: Path) -> list[Prompt]:
    prompts = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"line {line_number}: nonempty string id required")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str)
                or not m["content"]
                for m in messages
            )
        ):
            raise ValueError(f"line {line_number}: invalid messages")
        maximum = row.get("max_tokens", 128)
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError(f"line {line_number}: invalid max_tokens")
        expected = row.get("expected")
        if expected is not None and not isinstance(expected, str):
            raise ValueError(f"line {line_number}: expected must be text")
        prompts.append(Prompt(row["id"], messages, maximum, expected))
    if (
        not prompts
        or len(prompts) > 100000
        or len({p.id for p in prompts}) != len(prompts)
    ):
        raise ValueError("provide 1-100000 prompts with distinct ids")
    return prompts


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation (the method is recorded in the output)."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def normalize_answer(value: str) -> str:
    return " ".join(value.casefold().split())


class InvalidResponse(ValueError):
    pass


def parse_json_response(
    body: Any, config: Config
) -> tuple[str, bool | None, int | None]:
    if not isinstance(body, dict):
        raise InvalidResponse("JSON object required")
    if config.mode.startswith("gateway"):
        content, cached = body.get("content"), body.get("cached")
        if type(cached) is not bool:
            raise InvalidResponse("gateway cached must be boolean")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise InvalidResponse("gateway model missing")
        if config.model and body["model"] != config.model:
            raise InvalidResponse(
                "gateway model differs from requested benchmark model"
            )
    else:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponse("missing choices/message/content") from exc
        cached = None
    if not isinstance(content, str) or not content.strip():
        raise InvalidResponse("nonempty text content required")
    usage = body.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise InvalidResponse("invalid completion_tokens")
    return content, cached, tokens


async def parse_stream(
    response: httpx.Response, started: float
) -> tuple[str, float, int | None]:
    content = []
    first_content = None
    tokens = None
    finished = False
    async for line in response.aiter_lines():
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            finished = True
            break
        try:
            body = json.loads(data)
        except json.JSONDecodeError as exc:
            raise InvalidResponse("malformed SSE data JSON") from exc
        if not isinstance(body, dict):
            raise InvalidResponse("SSE data object required")
        usage = body.get("usage")
        if isinstance(usage, dict) and "completion_tokens" in usage:
            tokens = usage["completion_tokens"]
            if type(tokens) is not int or tokens < 0:
                raise InvalidResponse("invalid stream completion_tokens")
        choices = body.get("choices", [])
        if not isinstance(choices, list):
            raise InvalidResponse("invalid SSE choices")
        if not choices:  # usage-only final chunk
            continue
        if not isinstance(choices[0], dict):
            raise InvalidResponse("invalid SSE choice")
        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            raise InvalidResponse("invalid SSE delta")
        piece = delta.get("content")
        if piece is not None and not isinstance(piece, str):
            raise InvalidResponse("stream content must be text")
        if piece:
            if first_content is None:
                first_content = (time.perf_counter() - started) * 1000
            content.append(piece)
        if choices[0].get("finish_reason") is not None:
            finished = True
    if first_content is None or not "".join(content).strip() or not finished:
        raise InvalidResponse("incomplete stream or no content token")
    return "".join(content), first_content, tokens


async def run_benchmark(
    config: Config,
    prompts: list[Prompt],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[dict], dict]:
    """No automatic retries by default. Each attempted request remains in raw data.

    Bounded workers and queue prevent unlimited outstanding tasks. Queue delay
    and event-loop scheduling lag are load-generator metrics, not server latency.
    """
    config.validate()
    if not prompts:
        raise ValueError("empty workload")
    if config.mode == "gateway-cold" and config.requests > len(prompts):
        raise ValueError("cold-cache measurement cannot reuse prompts within a run")
    if config.mode == "gateway-cold":
        identities = [
            json.dumps(
                {"messages": p.messages, "max_tokens": p.max_tokens}, sort_keys=True
            )
            for p in prompts[: config.requests]
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("cold-cache workload has duplicate request payloads")
    records: list[dict] = []
    prime_content: dict[str, str] = {}
    loop_lags: list[float] = []
    inflight = peak_inflight = 0
    headers = (
        {"X-Cache-Namespace": config.namespace()} if config.cache_namespace else {}
    )
    limits = httpx.Limits(
        max_connections=config.concurrency, max_keepalive_connections=config.concurrency
    )
    async with httpx.AsyncClient(
        timeout=config.timeout_seconds,
        transport=transport,
        limits=limits,
        follow_redirects=False,
    ) as client:

        async def attempt(
            prompt: Prompt,
            index: int,
            phase: str,
            offered_at: float,
            attempt_number: int,
        ) -> dict:
            nonlocal inflight, peak_inflight
            payload = {
                "messages": prompt.messages,
                "max_tokens": prompt.max_tokens,
                "temperature": config.temperature,
                "seed": config.seed,
            }
            if config.mode.startswith("direct"):
                payload["model"] = config.model
            else:
                payload["use_cache"] = config.mode != "gateway-bypass"
            if config.mode == "direct-stream":
                payload.update(stream=True, stream_options={"include_usage": True})
            started = time.perf_counter()
            row = {
                "run_id": config.run_id,
                "phase": phase,
                "mode": config.mode,
                "request_index": index,
                "prompt_id": prompt.id,
                "attempt": attempt_number,
                "payload_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest(),
                "load_generator_wait_ms": (started - offered_at) * 1000,
                "e2e_ms": None,
                "ttft_ms": None,
                "http_status": None,
                "success": False,
                "error_type": None,
                "error": None,
                "cached": None,
                "completion_tokens": None,
                "content_sha256": None,
                "quality_correct": None,
                "has_reference": prompt.expected is not None,
            }
            inflight += 1
            peak_inflight = max(peak_inflight, inflight)
            try:
                if config.mode == "direct-stream":
                    async with client.stream(
                        "POST", config.url, json=payload, headers=headers
                    ) as response:
                        row["http_status"] = response.status_code
                        if response.status_code != 200:
                            raise InvalidResponse(f"HTTP {response.status_code}")
                        content, row["ttft_ms"], tokens = await parse_stream(
                            response, started
                        )
                        cached = None
                else:
                    response = await client.post(
                        config.url, json=payload, headers=headers
                    )
                    row["http_status"] = response.status_code
                    if response.status_code != 200:
                        raise InvalidResponse(f"HTTP {response.status_code}")
                    content, cached, tokens = parse_json_response(
                        response.json(), config
                    )
                row.update(
                    cached=cached,
                    completion_tokens=tokens,
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                )
                if phase == "measure":
                    if (
                        config.mode in ("gateway-bypass", "gateway-cold")
                        and cached is not False
                    ):
                        raise InvalidResponse(
                            "unexpected cache hit in bypass/cold measurement"
                        )
                    if config.mode == "gateway-warm":
                        if cached is not True:
                            raise InvalidResponse(
                                "expected warm cache hit; namespace/TTL/cache unavailable"
                            )
                        if content != prime_content[prompt.id]:
                            raise InvalidResponse(
                                "cache replay differs from primed response"
                            )
                elif config.mode == "gateway-warm":
                    prime_content[prompt.id] = content
                if prompt.expected is not None:
                    row["quality_correct"] = normalize_answer(
                        content
                    ) == normalize_answer(prompt.expected)
                row["success"] = True
            except (
                httpx.HTTPError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ) as exc:
                row.update(error_type=type(exc).__name__, error=str(exc))
            finally:
                row["e2e_ms"] = (time.perf_counter() - started) * 1000
                inflight -= 1
            records.append(row)
            return row

        # Prime outside timed measurement, one complete response per workload id.
        if config.mode == "gateway-warm":
            for index, prompt in enumerate(prompts):
                row = await attempt(prompt, index, "prime", time.perf_counter(), 1)
                if not row["success"]:
                    return records, {
                        "schema_version": 1,
                        "config": asdict(config),
                        "valid_run": False,
                        "validation_errors": [
                            f"warm-cache priming failed for {prompt.id}: {row['error']}"
                        ],
                        "logical_requests": 0,
                        "attempts": 0,
                        "successful_requests": 0,
                        "failed_requests": 0,
                        "failed_priming_requests": 1,
                    }

        queue: asyncio.Queue = asyncio.Queue(maxsize=config.concurrency)
        stop_monitor = asyncio.Event()
        started_measurement = time.perf_counter()

        async def monitor() -> None:
            interval = 0.01
            planned = time.perf_counter() + interval
            while not stop_monitor.is_set():
                await asyncio.sleep(max(0, planned - time.perf_counter()))
                if stop_monitor.is_set():
                    return
                now = time.perf_counter()
                loop_lags.append(max(0, now - planned) * 1000)
                planned = now + interval

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                index, prompt, offered_at = item
                logical_started = time.perf_counter()
                for number in range(1, config.retries + 2):
                    before_attempt = time.perf_counter()
                    row = await attempt(prompt, index, "measure", offered_at, number)
                    row["logical_e2e_ms"] = (
                        time.perf_counter() - logical_started
                    ) * 1000
                    row["logical_ttft_ms"] = (
                        (before_attempt - logical_started) * 1000 + row["ttft_ms"]
                        if row["ttft_ms"] is not None
                        else None
                    )
                    if row["success"]:
                        break
                    # Invalid data/cache assertions are not transient transport failures.
                    if row["error_type"] not in (
                        "ConnectError",
                        "ReadError",
                        "ReadTimeout",
                        "ConnectTimeout",
                        "PoolTimeout",
                        "WriteError",
                        "WriteTimeout",
                        "RemoteProtocolError",
                    ):
                        break
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(config.concurrency)]
        monitor_task = asyncio.create_task(monitor())
        for index in range(config.requests):
            planned = (
                started_measurement + index / config.arrival_rate
                if config.arrival_rate
                else time.perf_counter()
            )
            if config.arrival_rate:
                await asyncio.sleep(max(0, planned - time.perf_counter()))
            await queue.put((index, prompts[index % len(prompts)], planned))
        await queue.join()
        elapsed = time.perf_counter() - started_measurement
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        stop_monitor.set()
        await monitor_task
    measured = [r for r in records if r["phase"] == "measure"]
    final = {r["request_index"]: r for r in measured}
    successes = [r for r in final.values() if r["success"]]
    labeled = [r for r in final.values() if r["has_reference"]]
    known_tokens = [
        r["completion_tokens"] for r in successes if r["completion_tokens"] is not None
    ]
    validations = sorted(
        {
            r["error"]
            for r in measured
            if r["error_type"] == "InvalidResponse"
            and ("cache" in r["error"] or "benchmark model" in r["error"])
        }
    )
    summary = {
        "schema_version": 1,
        "config": asdict(config),
        "valid_run": not validations,
        "validation_errors": validations,
        "workload_sha256": hashlib.sha256(
            json.dumps([asdict(p) for p in prompts], sort_keys=True).encode()
        ).hexdigest(),
        "workload_unique_prompts": len(prompts),
        "logical_requests": len(final),
        "attempts": len(measured),
        "retry_attempts": len(measured) - len(final),
        "failed_attempts": sum(not r["success"] for r in measured),
        "successful_requests": len(successes),
        "failed_requests": len(final) - len(successes),
        "error_rate": (len(final) - len(successes)) / len(final),
        "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(successes) / elapsed,
        "peak_inflight": peak_inflight,
        "max_configured_inflight": config.concurrency,
        "success_e2e_ms": distribution([r["logical_e2e_ms"] for r in successes]),
        "all_final_e2e_ms": distribution([r["logical_e2e_ms"] for r in final.values()]),
        "attempt_e2e_ms": distribution([r["e2e_ms"] for r in measured]),
        "success_ttft_ms": distribution(
            [
                r["logical_ttft_ms"]
                for r in successes
                if r["logical_ttft_ms"] is not None
            ]
        ),
        "load_generator_wait_ms": distribution(
            [r["load_generator_wait_ms"] for r in measured]
        ),
        "load_generator_event_loop_lag_ms": distribution(loop_lags),
        "cache_hits": sum(r["cached"] is True for r in successes),
        "output_tokens_per_second": sum(known_tokens) / elapsed
        if len(known_tokens) == len(successes) and successes
        else None,
        "reference_requests": len(labeled),
        "reference_correct_requests": sum(
            r["success"] and r["quality_correct"] is True for r in labeled
        ),
        "reference_accuracy_including_failures": sum(
            r["success"] and r["quality_correct"] is True for r in labeled
        )
        / len(labeled)
        if labeled
        else None,
        "notes": [
            "Percentiles use linear interpolation; small samples do not establish stable tails.",
            "Client E2E includes HTTP connection/pool/network/decode time; it does not isolate server queue or GPU time.",
            "Load-generator wait and event-loop lag are separate from request E2E.",
            "Successful-request E2E/TTFT include prior failed attempts when retries are explicitly enabled; raw e2e_ms is per attempt.",
            "Warm-cache replay throughput is not model token-generation throughput.",
            "JSON gateway/direct modes have no observable TTFT; absent token counts are not estimated.",
            "Reference scores are workload smoke checks; repeated prompts are not independent quality evidence.",
        ],
    }
    if config.mode == "gateway-warm":
        summary["output_tokens_per_second"] = None
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full POST endpoint URL")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Operator-chosen fresh id; reusing cold ids is detected as cache hits",
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSONL: id, messages, max_tokens, optional expected",
    )
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--cache-namespace", default="")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Optional offered req/s; a bounded client queue may backpressure",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw_path = Path(str(args.output_prefix) + ".jsonl")
    summary_path = Path(str(args.output_prefix) + ".summary.json")
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError(
            "refusing to overwrite benchmark artifacts; choose a fresh output prefix"
        )
    prompts = load_prompts(args.prompts)
    config = Config(
        **{k: v for k, v in vars(args).items() if k not in ("prompts", "output_prefix")}
    )
    records, summary = asyncio.run(run_benchmark(config, prompts))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "raw": str(raw_path),
                "summary": str(summary_path),
                "failed_requests": summary["failed_requests"],
            }
        )
    )
    if summary["failed_requests"] or not summary["valid_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
