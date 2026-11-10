"""Run the real gateway and private Redis against a CPU-only HTTP fixture.

This is a protocol/cache/load-harness integration experiment, NOT LLM serving,
GPU inference, model accuracy, production traffic, or a vLLM benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from redis.exceptions import RedisError

from benchmarks.serving_benchmark import Config, load_prompts, run_benchmark

MODEL = "cpu-reference-fixture-v1"
NOMINAL_DELAY_SECONDS = 0.020
BASE_SOURCE_COMMIT = "69b66f63fc8d91afc3e6056b79c27b8a3de7942f"
ROOT = Path(__file__).resolve().parents[1]
reference_app = FastAPI(title="CPU reference fixture — not a language model")
_state = {"mode": "normal", "calls": 0, "active": 0, "peak_active": 0}


def fixture_content(messages: list[dict]) -> str:
    """Deterministic protocol sentinel; no learned model or tokenizer is involved."""
    encoded = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
    return "reference:" + hashlib.sha256(encoded).hexdigest()[:24]


@reference_app.get("/v1/models")
async def models():
    if _state["mode"] == "unhealthy":
        return JSONResponse(status_code=503, content={"error": "synthetic unready fixture"})
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@reference_app.get("/_fixture/stats")
async def stats():
    return dict(_state)


@reference_app.post("/_fixture/control")
async def control(request: Request):
    body = await request.json()
    if _state["active"]:
        return JSONResponse(status_code=409, content={"error": "fixture busy"})
    if body.get("mode", "normal") not in ("normal", "non-json", "timeout", "unhealthy"):
        return JSONResponse(status_code=400, content={"error": "unknown fixture mode"})
    _state["mode"] = body.get("mode", "normal")
    if body.get("reset"):
        _state.update(calls=0, active=0, peak_active=0)
    return dict(_state)


@reference_app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    if body.get("model") != MODEL or body.get("stream", False):
        return JSONResponse(status_code=400, content={"error": "fixture model/stream mismatch"})
    _state["calls"] += 1
    _state["active"] += 1
    _state["peak_active"] = max(_state["peak_active"], _state["active"])
    try:
        await asyncio.sleep(2 if _state["mode"] == "timeout" else NOMINAL_DELAY_SECONDS)
        if _state["mode"] == "non-json":
            return PlainTextResponse("synthetic non-JSON backend response", status_code=200)
        content = fixture_content(body["messages"])
        return {
            "id": "cpu-reference",
            "object": "chat.completion",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    finally:
        _state["active"] -= 1


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def source_hashes() -> dict:
    paths = sorted((ROOT / "app").glob("*.py")) + [
        ROOT / "benchmarks/serving_benchmark.py",
        ROOT / "benchmarks/reference_stack.py",
        ROOT / "benchmarks/fixtures/prompts.jsonl",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def environment(redis_info: dict) -> dict:
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unavailable",
    )
    limits = {}
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        limits[name] = path.read_text().strip() if path.exists() else "unavailable"
    return {
        "experiment_kind": "CPU HTTP fixture integration; not LLM or GPU performance",
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu,
        "logical_cpu_count": os.cpu_count(),
        "cgroup_limits": limits,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("httpx", "fastapi", "uvicorn", "redis", "redislite")
        },
        "redis": {
            key: redis_info.get(key)
            for key in ("redis_version", "redis_mode", "arch_bits", "multiplexing_api")
        },
        "redis_protocol": "RESP2 via redis-py; loopback TCP; actual bundled Redis server",
        "model": MODEL,
        "backend_implementation": "deterministic SHA-256 text sentinel; no tokenizer/model",
        "nominal_backend_delay_ms": NOMINAL_DELAY_SECONDS * 1000,
        "realized_delay_note": "asyncio scheduling can exceed nominal delay; measured client latency is recorded",
        "upstream_source_commit": BASE_SOURCE_COMMIT,
        "source_sha256": source_hashes(),
        "load_generator_location": "same shared host as both uvicorn processes and Redis; loopback HTTP",
        "backend_workers": 1,
        "gateway_workers": 1,
        "generation": {"temperature": 0, "seed": 0},
        "limitations": [
            "No GPU, no vLLM, no trained model, no model token throughput or TTFT.",
            "Synthetic deterministic text is a protocol sentinel, not held-out model quality.",
            "The four modes run separately; within each concurrency/replicate their order rotates.",
            "Backend/service warm-up is separate from response-cache priming.",
            "Short tests on a shared CPU host do not establish production SLOs.",
        ],
    }


class Server:
    def __init__(self, app: str, env: dict, log_path: Path):
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(2048)
        self.port = self.socket.getsockname()[1]
        self.log = log_path.open("x")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                app,
                "--fd",
                str(self.socket.fileno()),
                "--no-access-log",
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            pass_fds=(self.socket.fileno(),),
        )
        self.socket.close()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log.close()


class RedisServer:
    """Foreground bundled Redis using TCP where Unix sockets are unavailable."""

    def __init__(self, temporary: Path, output: Path):
        from redislite.client import __redis_executable__

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.log = (output / "redis.log").open("x")
        self.process = subprocess.Popen(
            [
                __redis_executable__,
                "--bind",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--daemonize",
                "no",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(temporary),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.client = redis.Redis(
            host="127.0.0.1", port=self.port, socket_timeout=0.5, socket_connect_timeout=0.5
        )

    async def ready(self):
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("bundled TCP Redis exited; see redis.log")
            try:
                if self.client.ping():
                    return
            except RedisError:
                pass
            await asyncio.sleep(0.05)
        raise TimeoutError("bundled TCP Redis did not become ready in 5 seconds")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.client.close()
        self.log.close()


async def wait_ready(url: str, process, *, seconds=15):
    deadline = time.perf_counter() + seconds
    async with httpx.AsyncClient(timeout=0.5, trust_env=False) as client:
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited before readiness: {url}")
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
    raise TimeoutError(f"server not ready within {seconds}s: {url}")


def aggregates(summaries: list[dict]) -> list[dict]:
    rows = []
    for mode in ("direct-json", "gateway-bypass", "gateway-cold", "gateway-warm"):
        for concurrency in sorted({s["config"]["concurrency"] for s in summaries}):
            selected = [
                s
                for s in summaries
                if s["config"]["mode"] == mode and s["config"]["concurrency"] == concurrency
            ]
            rows.append(
                {
                    "mode": mode,
                    "concurrency": concurrency,
                    "replicates": len(selected),
                    "successful_requests": sum(s["successful_requests"] for s in selected),
                    "failed_requests": sum(s["failed_requests"] for s in selected),
                    "cache_hits": sum(s["cache_hits"] for s in selected),
                    "median_of_run_p50_ms": statistics.median(
                        s["success_e2e_ms"]["p50"] for s in selected
                    ),
                    "median_of_run_p95_ms": statistics.median(
                        s["success_e2e_ms"]["p95"] for s in selected
                    ),
                    "median_run_success_req_s": statistics.median(
                        s["successful_requests_per_second"] for s in selected
                    ),
                    "max_event_loop_lag_ms": max(
                        s["load_generator_event_loop_lag_ms"]["max"] or 0 for s in selected
                    ),
                }
            )
    return rows


async def execute(output: Path, request_count: int, concurrencies: list[int], replicates: int):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    prompts = load_prompts(ROOT / "benchmarks/fixtures/prompts.jsonl")
    if request_count > len(prompts):
        raise ValueError("request_count exceeds unique reference fixture payloads")
    prompts = prompts[:request_count]
    env = dict(os.environ)
    # All fixture traffic is explicitly loopback. Do not send local benchmark
    # traffic through a user's remote proxy or require SOCKS extras to reach it.
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(name, None)
    servers = []
    redis_server = None
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="llmgateway-reference-") as temporary:
            redis_server = RedisServer(Path(temporary), output)
            await redis_server.ready()
            redis_instance = redis_server.client
            reference = Server(
                "benchmarks.reference_stack:reference_app", env, output / "reference.log"
            )
            servers.append(reference)
            reference_base = f"http://127.0.0.1:{reference.port}"
            model_info = await wait_ready(reference_base + "/v1/models", reference.process)
            if model_info["data"][0]["id"] != MODEL:
                raise AssertionError("reference model identity mismatch")
            env.update(
                LLM_BACKEND="selfhosted",
                VLLM_URL=reference_base + "/v1",
                VLLM_MODEL=MODEL,
                REDIS_URL=f"redis://127.0.0.1:{redis_server.port}/0",
                REQUEST_TIMEOUT="0.75",
                CACHE_TIMEOUT="0.15",
                READINESS_TIMEOUT="0.5",
                CACHE_TTL="3600",
                MODEL_REVISION="reference-fixture-v1",
                INFERENCE_DTYPE="not-applicable-cpu-http-fixture",
            )
            gateway = Server("app.main:app", env, output / "gateway.log")
            servers.append(gateway)
            gateway_base = f"http://127.0.0.1:{gateway.port}"
            await wait_ready(gateway_base + "/ready", gateway.process)
            metadata = environment(redis_instance.info())
            metadata["http_urls"] = {"reference": reference_base, "gateway": gateway_base}
            write_json(output / "environment.json", metadata)
            summaries = []
            modes = ["direct-json", "gateway-bypass", "gateway-cold", "gateway-warm"]
            async with httpx.AsyncClient(timeout=4, trust_env=False) as control_client:
                # Establish the HTTP paths before measured load. Cache bypass prevents priming.
                for _ in range(3):
                    response = await control_client.post(
                        gateway_base + "/v1/chat",
                        json={
                            "messages": prompts[0].messages,
                            "max_tokens": prompts[0].max_tokens,
                            "temperature": 0,
                            "seed": 0,
                            "use_cache": False,
                        },
                    )
                    response.raise_for_status()
                for replicate in range(1, replicates + 1):
                    for concurrency in concurrencies:
                        offset = (replicate - 1 + concurrencies.index(concurrency)) % len(modes)
                        for mode in modes[offset:] + modes[:offset]:
                            run_id = f"r{replicate}-c{concurrency}-{mode}"
                            print("RUN " + run_id, flush=True)
                            reset = await control_client.post(
                                reference_base + "/_fixture/control", json={"reset": True}
                            )
                            reset.raise_for_status()
                            config = Config(
                                url=reference_base + "/v1/chat/completions"
                                if mode.startswith("direct")
                                else gateway_base + "/v1/chat",
                                mode=mode,
                                run_id=run_id,
                                model=MODEL,
                                requests=request_count,
                                concurrency=concurrency,
                                retries=0,
                                timeout_seconds=4,
                                cache_namespace="cpu-reference-20261110",
                                temperature=0,
                                seed=0,
                            )
                            records, summary = await asyncio.wait_for(
                                run_benchmark(
                                    config,
                                    prompts,
                                    transport=httpx.AsyncHTTPTransport(trust_env=False),
                                ),
                                timeout=60,
                            )
                            backend_stats = (
                                await control_client.get(reference_base + "/_fixture/stats")
                            ).json()
                            summary["reference_backend_counters"] = backend_stats
                            # Preserve raw evidence before checking acceptance criteria.
                            with (output / (run_id + ".jsonl")).open("x") as handle:
                                for row in records:
                                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                            errors = []
                            if (
                                not summary["valid_run"]
                                or summary["failed_requests"]
                                or summary["logical_requests"] != request_count
                            ):
                                errors.append("run validity/count/error check failed")
                            if summary["reference_accuracy_including_failures"] != 1:
                                errors.append(
                                    "fixture protocol sentinel mismatch; not model quality"
                                )
                            if summary["peak_inflight"] > concurrency:
                                errors.append("configured client concurrency exceeded")
                            if backend_stats["calls"] != request_count:
                                errors.append(
                                    "backend-call count differs from direct/bypass/cold or warm-prime expectation"
                                )
                            measured = [r for r in records if r["phase"] == "measure"]
                            expected_hash = {
                                p.id: hashlib.sha256(p.expected.encode()).hexdigest()
                                for p in prompts
                            }
                            if any(
                                r["content_sha256"] != expected_hash[r["prompt_id"]]
                                for r in measured
                            ):
                                errors.append("cross-mode content hash mismatch")
                            if mode == "gateway-warm" and any(
                                r["cached"] is not False for r in records if r["phase"] == "prime"
                            ):
                                errors.append("fresh warm namespace unexpectedly already primed")
                            summary["reference_acceptance_errors"] = errors
                            write_json(output / (run_id + ".summary.json"), summary)
                            summaries.append(summary)
                            if errors:
                                raise AssertionError(f"{run_id}: {errors}")
                write_json(output / "aggregate.json", aggregates(summaries))
                faults = []
                body = {
                    "messages": prompts[0].messages,
                    "max_tokens": 8,
                    "temperature": 0,
                    "seed": 0,
                    "use_cache": False,
                }
                for fault, expected in (("non-json", 502), ("timeout", 503), ("unhealthy", 503)):
                    result = await control_client.post(
                        reference_base + "/_fixture/control", json={"mode": fault}
                    )
                    result.raise_for_status()
                    begin = time.perf_counter()
                    response = (
                        await control_client.get(gateway_base + "/ready")
                        if fault == "unhealthy"
                        else await control_client.post(gateway_base + "/v1/chat", json=body)
                    )
                    live = await control_client.get(gateway_base + "/health")
                    fault_result = {
                        "fault": fault,
                        "expected_http_status": expected,
                        "observed_http_status": response.status_code,
                        "liveness_http_status": live.status_code,
                        "elapsed_ms": (time.perf_counter() - begin) * 1000,
                        "passed": response.status_code == expected and live.status_code == 200,
                    }
                    faults.append(fault_result)
                    if fault == "timeout":
                        # The timed-out backend coroutine is independent of the client;
                        # let it finish before the guarded control endpoint changes mode.
                        await asyncio.sleep(1.4)
                    if not fault_result["passed"]:
                        write_json(output / "fault_smoke.json", faults)
                        raise AssertionError(fault_result)
                result = await control_client.post(
                    reference_base + "/_fixture/control", json={"mode": "normal"}
                )
                result.raise_for_status()
                redis_instance.shutdown()
                body["use_cache"] = True
                begin = time.perf_counter()
                response = await control_client.post(
                    gateway_base + "/v1/chat",
                    json=body,
                    headers={"X-Cache-Namespace": "cpu-reference-redis-down"},
                )
                fault_result = {
                    "fault": "Redis server stopped",
                    "expected_http_status": 200,
                    "observed_http_status": response.status_code,
                    "cached": response.json().get("cached"),
                    "elapsed_ms": (time.perf_counter() - begin) * 1000,
                    "passed": response.status_code == 200
                    and response.json().get("cached") is False,
                }
                faults.append(fault_result)
                write_json(output / "fault_smoke.json", faults)
                if not fault_result["passed"]:
                    raise AssertionError(fault_result)
            final_hashes = source_hashes()
            completion = {
                "matrix_runs": len(summaries),
                "measured_requests": sum(s["logical_requests"] for s in summaries),
                "failed_requests": sum(s["failed_requests"] for s in summaries),
                "fault_checks_passed": all(f["passed"] for f in faults),
                "source_unchanged_during_run": final_hashes == metadata["source_sha256"],
                "elapsed_seconds": time.perf_counter() - started,
                "kind": "CPU reference fixture integration, not model/GPU performance",
            }
            write_json(output / "completion.json", completion)
            if not completion["source_unchanged_during_run"]:
                raise AssertionError("measured source changed during run")
            print(json.dumps(completion, sort_keys=True), flush=True)
    finally:
        for server in reversed(servers):
            server.close()
        if redis_server is not None:
            redis_server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    if (
        not 1 <= args.requests <= 200
        or not 1 <= args.replicates <= 10
        or not concurrencies
        or any(not 1 <= c <= 64 for c in concurrencies)
    ):
        raise ValueError("invalid bounded reference experiment settings")
    asyncio.run(execute(args.output_dir, args.requests, concurrencies, args.replicates))


if __name__ == "__main__":
    main()
