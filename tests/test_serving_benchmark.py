"""CPU/mock harness tests. These timings are never evidence of model performance."""

import asyncio
import json

import httpx
import pytest

from benchmarks.serving_benchmark import (
    Config,
    Prompt,
    load_prompts,
    percentile,
    run_benchmark,
)


def prompt(i=0, expected=None):
    return Prompt(str(i), [{"role": "user", "content": f"Return the number {i}."}], 8, expected)


def config(mode="gateway-bypass", **kwargs):
    return Config(url="http://cpu-fixture/chat", mode=mode, run_id="fixture", **kwargs)


def run(conf, prompts, handler):
    return asyncio.run(run_benchmark(conf, prompts, transport=httpx.MockTransport(handler)))


def test_bounded_concurrency_preserves_payload_and_never_invents_ttft():
    active = peak = 0
    received = []

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        received.append(json.loads(request.content))
        await asyncio.sleep(0.005)
        active -= 1
        return httpx.Response(200, json={"content": "0", "cached": False, "model": "fixture"})

    records, summary = run(config(requests=17, concurrency=3), [prompt()], handler)
    assert peak == summary["peak_inflight"] == 3
    assert summary["logical_requests"] == summary["successful_requests"] == 17
    assert all(r["ttft_ms"] is None for r in records)
    assert summary["success_ttft_ms"]["n"] == 0
    assert summary["output_tokens_per_second"] is None
    assert all(x["messages"] == prompt().messages and x["use_cache"] is False for x in received)
    assert all(r["e2e_ms"] >= 0 and r["load_generator_wait_ms"] >= 0 for r in records)


def test_cold_namespace_does_not_change_prompt_and_detects_existing_hits():
    seen = []

    async def handler(request):
        seen.append((request.headers["X-Cache-Namespace"], json.loads(request.content)))
        return httpx.Response(
            200, json={"content": "1", "cached": len(seen) == 2, "model": "fixture"}
        )

    records, summary = run(
        config("gateway-cold", requests=2, cache_namespace="isolated"),
        [prompt(1), prompt(2)],
        handler,
    )
    assert [r[1]["messages"] for r in seen] == [prompt(1).messages, prompt(2).messages]
    assert all(r[0] == "isolated/fixture/gateway-cold" for r in seen)
    assert summary["successful_requests"] == 1
    assert summary["valid_run"] is False
    assert records[1]["error_type"] == "InvalidResponse"


def test_cold_refuses_repeated_payload_even_with_different_ids():
    clone = Prompt("different-id", prompt().messages, 8)
    with pytest.raises(ValueError, match="duplicate request payloads"):
        run(
            config("gateway-cold", requests=2, cache_namespace="isolated"),
            [prompt(), clone],
            lambda r: None,
        )
    with pytest.raises(ValueError, match="reuse prompts"):
        run(
            config("gateway-cold", requests=2, cache_namespace="isolated"),
            [prompt()],
            lambda r: None,
        )


def test_warm_priming_is_separate_and_hit_content_is_verified():
    cache = {}

    async def handler(request):
        payload = json.loads(request.content)
        key = payload["messages"][0]["content"]
        hit = key in cache
        cache.setdefault(key, key[-2])
        return httpx.Response(200, json={"content": cache[key], "cached": hit, "model": "fixture"})

    records, summary = run(
        config("gateway-warm", requests=6, concurrency=2, cache_namespace="isolated"),
        [prompt(1, "1"), prompt(2, "2")],
        handler,
    )
    assert len([r for r in records if r["phase"] == "prime"]) == 2
    assert summary["logical_requests"] == 6 and summary["cache_hits"] == 6
    assert summary["reference_accuracy_including_failures"] == 1
    assert summary["output_tokens_per_second"] is None


def test_warm_content_change_or_cache_outage_invalidates_run():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "content": "first" if calls == 1 else "changed",
                "cached": calls > 1,
                "model": "fixture",
            },
        )

    records, summary = run(
        config("gateway-warm", requests=1, cache_namespace="isolated"),
        [prompt()],
        handler,
    )
    assert summary["valid_run"] is False
    assert summary["failed_requests"] == 1
    assert "replay differs" in records[-1]["error"]


def test_priming_failure_retains_raw_failure_and_runs_no_load():
    records, summary = run(
        config("gateway-warm", requests=5, cache_namespace="isolated"),
        [prompt()],
        lambda r: httpx.Response(503),
    )
    assert summary["valid_run"] is False and summary["logical_requests"] == 0
    assert len(records) == 1 and records[0]["phase"] == "prime"
    assert records[0]["http_status"] == 503


def test_retry_keeps_first_timeout_in_raw_and_does_not_hide_failure():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("synthetic timeout")
        return httpx.Response(200, json={"content": "ok", "cached": False, "model": "fixture"})

    records, summary = run(config(requests=1, retries=1), [prompt()], handler)
    assert summary["attempts"] == 2 and summary["retry_attempts"] == 1
    assert summary["failed_attempts"] == 1 and summary["failed_requests"] == 0
    assert records[0]["error_type"] == "ReadTimeout" and records[1]["attempt"] == 2


def test_errors_and_malformed_200_are_not_successes_or_silently_retried():
    responses = iter(
        [
            httpx.Response(500),
            httpx.Response(200, text="not-json"),
            httpx.Response(200, json={"content": "ok", "cached": "false", "model": "m"}),
        ]
    )
    records, summary = run(config(requests=3, retries=2), [prompt()], lambda r: next(responses))
    assert summary["failed_requests"] == summary["attempts"] == 3
    assert summary["success_e2e_ms"]["p50"] is None
    assert all(r["error_type"] for r in records)


class TokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        await asyncio.sleep(0.02)
        yield b'data: {"choices":[{"delta":{"content":"four"}}]}\n\n'
        await asyncio.sleep(0.01)
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
        yield b"data: [DONE]\n\n"


def test_stream_ttft_waits_for_content_and_keeps_token_usage():
    records, summary = run(
        config("direct-stream", model="fixture", requests=1),
        [prompt(expected="FOUR")],
        lambda r: httpx.Response(200, stream=TokenStream()),
    )
    row = records[0]
    assert row["success"] and row["ttft_ms"] >= 15
    assert row["e2e_ms"] > row["ttft_ms"] and row["completion_tokens"] == 1
    assert summary["reference_accuracy_including_failures"] == 1
    assert summary["output_tokens_per_second"] > 0


def test_truncated_or_malformed_stream_is_a_recorded_error():
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    records, summary = run(
        config("direct-stream", model="fixture", requests=1),
        [prompt()],
        lambda r: httpx.Response(200, stream=Broken()),
    )
    assert summary["failed_requests"] == 1
    assert "incomplete stream" in records[0]["error"]


def test_references_count_request_errors_as_incorrect():
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(200, json={"content": "wrong", "cached": False, "model": "m"}),
        ]
    )
    _, summary = run(config(requests=2), [prompt(expected="0")], lambda r: next(responses))
    assert summary["reference_requests"] == 2
    assert summary["reference_accuracy_including_failures"] == 0


def test_workload_validation_and_quantile_definition(tmp_path):
    file = tmp_path / "prompts.jsonl"
    file.write_text(
        json.dumps({"id": "one", "messages": prompt().messages, "max_tokens": 8}) + "\n"
    )
    assert load_prompts(file)[0].id == "one"
    file.write_text(file.read_text() * 2)
    with pytest.raises(ValueError, match="distinct ids"):
        load_prompts(file)
    assert percentile([0, 10], 0.95) == 9.5
    assert percentile([], 0.5) is None


@pytest.mark.parametrize(
    "settings",
    [
        {"requests": 0},
        {"concurrency": 0},
        {"timeout_seconds": float("nan")},
        {"retries": 4},
        {"arrival_rate": 0},
    ],
)
def test_invalid_load_settings_fail_before_request(settings):
    with pytest.raises(ValueError):
        config(**settings).validate()


def test_explicit_sse_error_after_partial_content_is_not_a_success():
    class ErrorAfterText(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            yield b'data: {"error":{"message":"upstream failure"}}\n\n'
            yield b"data: [DONE]\n\n"

    records, summary = run(
        config("direct-stream", model="fixture", requests=1, retries=2),
        [prompt()],
        lambda r: httpx.Response(200, stream=ErrorAfterText()),
    )
    assert summary["successful_requests"] == 0 and summary["failed_requests"] == 1
    assert summary["attempts"] == 1  # A declared server error is not retried as transport I/O.
    assert records[0]["error_type"] == "InvalidResponse"
    assert "SSE error" in records[0]["error"]


@pytest.mark.parametrize("streaming", [False, True])
def test_absolute_attempt_deadline_bounds_json_and_continuous_streams(streaming):
    class Heartbeats(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            while True:
                yield b": heartbeat\n\n"
                await asyncio.sleep(0.001)

        async def aclose(self):
            self.closed = True

    stream = Heartbeats()

    async def handler(request):
        if streaming:
            return httpx.Response(200, stream=stream)
        await asyncio.Event().wait()

    async def exercise():
        conf = config(
            "direct-stream" if streaming else "direct-json",
            model="fixture",
            requests=1,
            timeout_seconds=0.02,
        )
        # This outer bound detects an absent per-attempt deadline rather than
        # relying on MockTransport to implement HTTPX network inactivity timers.
        return await asyncio.wait_for(
            run_benchmark(conf, [prompt()], transport=httpx.MockTransport(handler)),
            timeout=0.5,
        )

    records, summary = asyncio.run(exercise())
    assert summary["failed_requests"] == 1 and summary["successful_requests"] == 0
    assert records[0]["error_type"] == "TimeoutError"
    assert "deadline exceeded" in records[0]["error"]
    if streaming:
        assert stream.closed


def test_retries_do_not_turn_failed_request_time_into_queue_wait():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return httpx.Response(200, json={"content": "ok", "cached": False, "model": "fixture"})

    records, summary = run(config(requests=1, retries=1, timeout_seconds=0.02), [prompt()], handler)
    assert summary["successful_requests"] == 1 and summary["failed_attempts"] == 1
    assert records[0]["error_type"] == "TimeoutError"
    assert records[0]["load_generator_wait_ms"] == records[1]["load_generator_wait_ms"]
    assert summary["load_generator_wait_ms"]["n"] == 1
    assert records[-1]["logical_e2e_ms"] >= records[0]["e2e_ms"]


def test_existing_warm_namespace_is_rejected_before_measured_requests():
    records, summary = run(
        config("gateway-warm", requests=3, cache_namespace="reused"),
        [prompt()],
        lambda r: httpx.Response(
            200, json={"content": "old cached response", "cached": True, "model": "fixture"}
        ),
    )
    assert summary["valid_run"] is False and summary["logical_requests"] == 0
    assert summary["failed_priming_requests"] == 1
    assert len(records) == 1 and records[0]["phase"] == "prime"
    assert "already contains" in records[0]["error"]


def test_workload_extra_message_keys_cannot_change_direct_payload_only(tmp_path):
    file = tmp_path / "prompts.jsonl"
    file.write_text(
        json.dumps(
            {
                "id": "one",
                "messages": [{"role": "user", "content": "hello", "name": "extra"}],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="invalid messages"):
        load_prompts(file)
