"""Regenerate the CPU reference report and figure from preserved run evidence.

This reads the archived summaries for latency/throughput and independently
checks request counts, cache outcomes, content hashes, and source provenance.
It never starts a model, reruns measurements, or estimates missing metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "benchmarks/runs/cpu-reference-20261110"
MODES = ("direct-json", "gateway-bypass", "gateway-cold", "gateway-warm")
LABELS = {
    "direct-json": "Direct HTTP fixture",
    "gateway-bypass": "Gateway, cache bypass",
    "gateway-cold": "Gateway, cold cache",
    "gateway-warm": "Gateway, verified warm cache",
}
COLORS = ("#34495e", "#0072b2", "#d55e00", "#009e73")
MARKERS = ("o", "s", "^", "D")
FENCE = chr(96) * 3
TICK = chr(96)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value, description: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description}: expected a number",
    )
    require(math.isfinite(value), f"{description}: non-finite number")
    return float(value)


def close(actual, expected, description: str) -> None:
    require(
        math.isclose(finite(actual, description), expected, rel_tol=1e-9, abs_tol=1e-7),
        f"{description}: summary/aggregate disagreement",
    )


def load_evidence(run_dir: Path) -> dict:
    environment = read_json(run_dir / "environment.json")
    completion = read_json(run_dir / "completion.json")
    faults = read_json(run_dir / "fault_smoke.json")
    grouped = defaultdict(list)
    total_measured = total_priming = total_failures = total_cache_hits = 0
    reference_content = {}
    workload_hashes = set()

    # Validate against the measured snapshot, not the subsequently edited tree.
    for name, expected in environment["source_sha256"].items():
        actual = hashlib.sha256((run_dir / "source_snapshot" / name).read_bytes()).hexdigest()
        require(actual == expected, f"Archived source hash mismatch: {name}")

    summary_paths = sorted(run_dir.glob("*.summary.json"))
    require(bool(summary_paths), "No per-run summaries found")
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        config = summary["config"]
        run_id = config["run_id"]
        mode = config["mode"]
        require(mode in MODES, f"{run_id}: unsupported mode")
        require(summary_path.name == run_id + ".summary.json", f"{run_id}: filename mismatch")
        require(
            config["retries"] == 0, "This report requires the archived zero-retry reference matrix"
        )
        require(
            summary["valid_run"]
            and not summary["validation_errors"]
            and not summary["reference_acceptance_errors"],
            f"{run_id}: run did not pass its recorded validity checks",
        )
        records = [
            json.loads(line)
            for line in (run_dir / (run_id + ".jsonl")).read_text().splitlines()
            if line.strip()
        ]
        measured = [r for r in records if r["phase"] == "measure"]
        prime = [r for r in records if r["phase"] == "prime"]
        require(len(measured) + len(prime) == len(records), f"{run_id}: unknown request phase")
        require(
            len(measured) == config["requests"] == summary["logical_requests"],
            f"{run_id}: measured request count mismatch",
        )
        require(
            len({r["request_index"] for r in measured}) == len(measured),
            f"{run_id}: repeated logical request in zero-retry matrix",
        )
        successes = sum(r["success"] is True for r in measured)
        failures = len(measured) - successes
        hits = sum(r["cached"] is True for r in measured)
        require(
            successes == summary["successful_requests"]
            and failures == summary["failed_requests"]
            and hits == summary["cache_hits"],
            f"{run_id}: raw records disagree with count summary",
        )
        require(
            summary["success_e2e_ms"]["n"] == successes,
            f"{run_id}: successful latency sample count mismatch",
        )
        close(summary["error_rate"], failures / len(measured), f"{run_id} error rate")
        close(
            summary["successful_requests_per_second"],
            successes / summary["elapsed_seconds"],
            f"{run_id} throughput",
        )
        require(
            summary["peak_inflight"] <= config["concurrency"],
            f"{run_id}: client concurrency bound exceeded",
        )
        require(all(r["attempt"] == 1 for r in records), f"{run_id}: unexpected retry")
        require(
            summary["output_tokens_per_second"] is None and summary["success_ttft_ms"]["n"] == 0,
            f"{run_id}: CPU JSON fixture unexpectedly reports model metrics",
        )

        if mode == "gateway-warm":
            require(
                len(prime) == len(measured)
                and all(r["success"] and r["cached"] is False for r in prime),
                f"{run_id}: priming was incomplete or unexpectedly cached",
            )
            require(hits == len(measured), f"{run_id}: warm replay was not all hits")
            primed_hashes = {r["prompt_id"]: r["content_sha256"] for r in prime}
            require(
                all(r["content_sha256"] == primed_hashes[r["prompt_id"]] for r in measured),
                f"{run_id}: replay content changed",
            )
        else:
            require(not prime and hits == 0, f"{run_id}: non-warm path has priming or cache hits")
            if mode.startswith("gateway"):
                require(
                    all(r["cached"] is False for r in measured),
                    f"{run_id}: missing explicit cache-miss result",
                )

        # Warm-mode calls are for priming; the other modes call the fixture
        # during measurement. The counter covers both phases.
        require(
            summary["reference_backend_counters"]["calls"] == config["requests"],
            f"{run_id}: fixture call count differs from path expectation",
        )
        for row in measured:
            require(
                row["has_reference"] and row["quality_correct"] is True,
                f"{run_id}: deterministic sentinel check failed",
            )
            previous = reference_content.setdefault(row["prompt_id"], row["content_sha256"])
            require(
                previous == row["content_sha256"],
                f"{run_id}: content sentinel differs across paths",
            )
        workload_hashes.add(summary["workload_sha256"])
        grouped[(mode, config["concurrency"])].append(summary)
        total_measured += len(measured)
        total_priming += len(prime)
        total_failures += failures
        total_cache_hits += hits

    require(len(workload_hashes) == 1, "Matrix does not use one fixed workload hash")
    concurrencies = sorted({key[1] for key in grouped})
    require(concurrencies == [1, 4, 8, 16], "This reference report expects concurrency 1, 4, 8, 16")
    rows = []
    for mode in MODES:
        for concurrency in concurrencies:
            selected = grouped[(mode, concurrency)]
            require(len(selected) == 3, f"{mode} C={concurrency}: expected three repetitions")
            require(len({s["config"]["run_id"] for s in selected}) == 3, "Duplicate run identifier")
            p50 = [finite(s["success_e2e_ms"]["p50"], "P50") for s in selected]
            p95 = [finite(s["success_e2e_ms"]["p95"], "P95") for s in selected]
            rates = [finite(s["successful_requests_per_second"], "request rate") for s in selected]
            rows.append(
                {
                    "mode": mode,
                    "concurrency": concurrency,
                    "replicates": 3,
                    "successful_requests": sum(s["successful_requests"] for s in selected),
                    "failed_requests": sum(s["failed_requests"] for s in selected),
                    "cache_hits": sum(s["cache_hits"] for s in selected),
                    "median_of_run_p50_ms": statistics.median(p50),
                    "median_of_run_p95_ms": statistics.median(p95),
                    "run_p95_min_ms": min(p95),
                    "run_p95_max_ms": max(p95),
                    "median_run_success_req_s": statistics.median(rates),
                    "run_rate_min": min(rates),
                    "run_rate_max": max(rates),
                    "max_event_loop_lag_ms": max(
                        s["load_generator_event_loop_lag_ms"]["max"] for s in selected
                    ),
                }
            )
    archived_aggregate = {
        (r["mode"], r["concurrency"]): r for r in read_json(run_dir / "aggregate.json")
    }
    require(set(archived_aggregate) == set(grouped), "Aggregate group mismatch")
    for row in rows:
        archived = archived_aggregate[(row["mode"], row["concurrency"])]
        for field in (
            "median_of_run_p50_ms",
            "median_of_run_p95_ms",
            "median_run_success_req_s",
            "max_event_loop_lag_ms",
        ):
            close(archived[field], row[field], f"aggregate {field}")
        for field in ("replicates", "successful_requests", "failed_requests", "cache_hits"):
            require(archived[field] == row[field], f"aggregate {field} mismatch")
    require(
        completion["matrix_runs"] == len(summary_paths)
        and completion["measured_requests"] == total_measured
        and completion["failed_requests"] == total_failures,
        "Completion totals disagree with request evidence",
    )
    require(completion["source_unchanged_during_run"], "Recorded source changed during run")
    require(len(faults) == 4, "Expected four separately recorded fault checks")
    require(
        completion["fault_checks_passed"] == all(f["passed"] for f in faults),
        "Completion/fault outcome mismatch",
    )
    return {
        "environment": environment,
        "completion": completion,
        "faults": faults,
        "rows": rows,
        "summaries": [s for selected in grouped.values() for s in selected],
        "measured": total_measured,
        "priming": total_priming,
        "failures": total_failures,
        "hits": total_cache_hits,
        "workload_hash": next(iter(workload_hashes)),
    }


def make_figure(evidence: dict, figure_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#52606d",
            "axes.linewidth": 0.7,
            "grid.color": "#dce2e7",
            "grid.linewidth": 0.6,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.73, wspace=0.25)
    for mode, color, marker in zip(MODES, COLORS, MARKERS):
        rows = [r for r in evidence["rows"] if r["mode"] == mode]
        x = [r["concurrency"] for r in rows]
        for ax, middle, lower, upper in (
            (axes[0], "median_of_run_p95_ms", "run_p95_min_ms", "run_p95_max_ms"),
            (axes[1], "median_run_success_req_s", "run_rate_min", "run_rate_max"),
        ):
            y = [r[middle] for r in rows]
            error = [[r[middle] - r[lower] for r in rows], [r[upper] - r[middle] for r in rows]]
            ax.errorbar(
                x,
                y,
                yerr=error,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.7,
                capsize=3,
                elinewidth=1,
                label=LABELS[mode],
            )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Configured client concurrency")
        ax.grid(axis="y")
        ax.set_ylim(bottom=0)
    axes[0].set_title("A   Successful request latency", loc="left", fontweight="bold")
    axes[0].set_ylabel("Client end-to-end P95 (ms)")
    axes[1].set_title("B   Successful request rate", loc="left", fontweight="bold")
    axes[1].set_ylabel("Successful measured requests / s")
    fig.suptitle(
        "CPU HTTP reference fixture — no LLM or GPU",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.905,
        "Nominal fixture delay: 20 ms • actual Redis over loopback TCP • all processes share one host",
        ha="left",
        fontsize=10,
        color="#3b4a54",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.79),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.6,
    )
    fig.text(
        0.075,
        0.062,
        "Markers: median of 3 run summaries. Bars: observed minimum–maximum across those 3 runs; not confidence intervals.",
        ha="left",
        fontsize=8.7,
        color="#3b4a54",
    )
    fig.text(
        0.075,
        0.025,
        "200 measured requests/run; warm-cache priming excluded. Cached replay is not model inference acceleration.",
        ha="left",
        fontsize=8.7,
        color="#3b4a54",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figure_path,
        dpi=220,
        metadata={
            "Title": "CPU HTTP reference gateway benchmark, 2026-11-10",
            "Description": "Measured middleware fixture results, not LLM/GPU performance. "
            "Median and min-max across three runs; priming excluded.",
            "Software": f"matplotlib {matplotlib.__version__}",
        },
    )
    plt.close(fig)


def link_from(report_path: Path, target: Path) -> str:
    import os

    return Path(os.path.relpath(target, report_path.parent)).as_posix()


def make_report(evidence: dict, run_dir: Path, report_path: Path, figure_path: Path) -> str:
    env = evidence["environment"]
    completion = evidence["completion"]
    raw = link_from(report_path, run_dir)
    figure = link_from(report_path, figure_path)
    bt = TICK
    lines = [
        "# Gateway benchmark results",
        "",
        "<!-- Generated by benchmarks/generate_report.py from preserved run evidence. -->",
        "",
        "**Measured 2026-11-10: CPU HTTP fixture integration, not LLM or GPU performance.**",
        (
            f"The preserved matrix contains **{completion['matrix_runs']:,} runs and "
            f"{evidence['measured']:,} measured requests**, with **{evidence['failures']:,} "
            f"failed measured requests**. Another **{evidence['priming']:,} successful "
            "warm-cache priming requests** are excluded from measured latency and throughput. "
            f"The {len(evidence['faults'])} controlled fault checks below are separate from this matrix."
        ),
        "",
        (
            "The upstream is a deterministic SHA-256 text sentinel with a nominal 20 ms "
            "asynchronous delay. It has no tokenizer, trained model, vLLM engine, or GPU. "
            "Its content checks establish protocol consistency, not language-model accuracy."
        ),
        "",
        f"![CPU fixture P95 latency and successful request rate]({figure})",
        "",
        (
            "Markers show the median of three per-run summaries; error bars span the "
            "**observed minimum–maximum of those three runs, not a confidence interval**. "
            "All services and the client share one host."
        ),
        "",
        "## Method and environment",
        "",
        (
            "- The same 200 unique fixed payloads, generation parameters, and order were "
            "used across four separately measured paths, at client concurrency 1, 4, 8, and 16."
        ),
        (
            "- Three repetitions were collected per path/concurrency. Path order rotated "
            "within each repetition/concurrency block; the runs were sequential."
        ),
        (
            "- Retries were disabled. This is a bounded-concurrency closed-loop workload, "
            "not an open-loop arrival-rate or sustained production-load study."
        ),
        (
            "- Service/HTTP warm-up was separate from response-cache priming. Health and "
            "readiness checks were outside the measured request matrix."
        ),
        (
            "- Cold-cache runs used fresh namespaces and distinct fixed payloads; measured "
            "responses were cache misses. Warm runs primed the same payloads, then required "
            "cache hits with content identical to the primed answers."
        ),
        (
            "- Client end-to-end latency includes HTTP, connection/pool, decoding, and "
            "scheduling effects. Percentiles use linear interpolation. Reported group values "
            "are medians of run summaries, not pooled request percentiles."
        ),
        "",
        "| Field | Recorded value |",
        "| --- | --- |",
        f"| CPU exposed by host | {env['cpu_model']} |",
        f"| Logical CPUs visible to process | {env['logical_cpu_count']} |",
        f"| Platform | {env['platform']} |",
        f"| Python | {env['python'].split(' (')[0]} |",
        f"| Gateway / fixture workers | {env['gateway_workers']} / {env['backend_workers']} |",
        f"| Fixture ID | {bt}{env['model']}{bt} |",
        (
            f"| Nominal fixture delay | {env['nominal_backend_delay_ms']:g} ms; "
            "not an observed service-time guarantee |"
        ),
        (
            f"| Redis server | {env['redis']['redis_version']}; standalone, "
            f"{env['redis']['arch_bits']}-bit, loopback TCP |"
        ),
        f"| Redis client protocol | {env['redis_protocol']} |",
        "| Client placement | Same shared host as gateway, fixture and Redis; loopback HTTP |",
        "| CPU / memory quota metadata | cgroup limits unavailable in recorded environment |",
        (
            f"| Full experiment wall time | {completion['elapsed_seconds']:.2f} s; "
            "includes priming, controls and fault checks, not a throughput denominator |"
        ),
        "",
        "Recorded measurement dependencies: "
        + ", ".join(
            f"{bt}{name}=={version}{bt}" for name, version in sorted(env["dependencies"].items())
        )
        + f". Figure generated with {bt}matplotlib=={matplotlib.__version__}{bt}.",
        "",
        "## All path/concurrency results",
        "",
        (
            "Every row summarizes three runs × 200 measured requests = 600 measured "
            "requests. Brackets contain the minimum and maximum **per-run** result. "
            "Successful request rate is successful measured requests divided by that run's "
            "measured elapsed time; priming is excluded."
        ),
        "",
        (
            "| Client C | Path | Median run P50 (ms) | Median run P95 [min–max] (ms) | "
            "Median successful req/s [min–max] | Failed / measured | Cache hits / measured |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in evidence["rows"]:
        n = row["successful_requests"] + row["failed_requests"]
        lines.append(
            f"| {row['concurrency']} | {bt}{row['mode']}{bt} | "
            f"{row['median_of_run_p50_ms']:.2f} | "
            f"{row['median_of_run_p95_ms']:.2f} "
            f"[{row['run_p95_min_ms']:.2f}–{row['run_p95_max_ms']:.2f}] | "
            f"{row['median_run_success_req_s']:.2f} "
            f"[{row['run_rate_min']:.2f}–{row['run_rate_max']:.2f}] | "
            f"{row['failed_requests']} / {n} | {row['cache_hits']} / {n} |"
        )
    warm16 = next(
        r for r in evidence["rows"] if r["mode"] == "gateway-warm" and r["concurrency"] == 16
    )
    cold16 = next(
        r for r in evidence["rows"] if r["mode"] == "gateway-cold" and r["concurrency"] == 16
    )
    lines += [
        "",
        (
            f"At concurrency 16, verified warm replay had median run P95 "
            f"**{warm16['median_of_run_p95_ms']:.2f} ms** "
            f"(run range {warm16['run_p95_min_ms']:.2f}–{warm16['run_p95_max_ms']:.2f}), "
            f"and median successful request rate **{warm16['median_run_success_req_s']:.2f} req/s**. "
            f"The cold-cache path had median run P95 **{cold16['median_of_run_p95_ms']:.2f} ms**. "
            "These are observations from different sequential runs on one shared CPU host. "
            "They are not a production SLO, an isolated estimate of gateway overhead, "
            "or evidence of faster model inference."
        ),
        "",
        (
            f"All {evidence['hits']:,} measured warm responses were verified cache hits. "
            "For each warm run the fixture counter recorded 200 calls during priming "
            "and no additional calls during measured replay; non-warm runs recorded "
            "200 calls during measurement. The report validates those counters and "
            "cross-path content sentinels against the preserved request files."
        ),
        "",
        "## Separate controlled fault checks",
        "",
        (
            "| Fault / operation | Expected HTTP | Observed HTTP | Liveness HTTP | "
            "Recorded interval (ms) | Outcome |"
        ),
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for fault in evidence["faults"]:
        live = str(fault.get("liveness_http_status", "not recorded"))
        outcome = "pass" if fault["passed"] else "FAIL"
        if "cached" in fault:
            outcome += f"; cached={str(fault['cached']).lower()}"
        lines.append(
            f"| {fault['fault']} | {fault['expected_http_status']} | "
            f"{fault['observed_http_status']} | {live} | "
            f"{fault['elapsed_ms']:.2f} | {outcome} |"
        )
    lines += [
        "",
        (
            "The unhealthy-backend check called readiness; non-JSON and timeout checks "
            "called the chat route. Their recorded intervals include the following "
            "liveness request in the archived driver. The Redis-down interval covers "
            "the chat request, which succeeded uncached. These four one-off intervals "
            "are not latency percentiles or strict deadline guarantees. Expected fault "
            "responses are excluded from the normal-workload failure count."
        ),
        "",
        "## Provenance and regeneration",
        "",
        (
            f"- [Environment and source hashes]({raw}/environment.json), "
            f"[completion totals]({raw}/completion.json), "
            f"[archived aggregate]({raw}/aggregate.json), "
            f"[fault evidence]({raw}/fault_smoke.json)."
        ),
        (
            f"- [Full raw-data archive](evidence/{run_dir.name}.tar.gz) includes "
            "request JSONL, logs, summaries, metadata, and the measured source snapshot. "
            "The repository retains summaries and provenance unpacked; extract the archive "
            "before regenerating from a fresh clone. Verify its checksum against the "
            "[SHA256SUMS](evidence/SHA256SUMS) before extraction."
        ),
        (
            f"- Extracted per-request evidence is in {bt}{raw}/r*.jsonl{bt}; each run has a "
            f"matching {bt}.summary.json{bt}. Raw records distinguish "
            f"{bt}prime{bt} and {bt}measure{bt}."
        ),
        (
            f"- Upstream source base: {bt}{env['upstream_source_commit']}{bt}. "
            "The measurements include working-tree changes; that upstream commit alone "
            "does not identify the measured implementation."
        ),
        (
            f"- [Exact measured source snapshot]({raw}/source_snapshot/) contains the "
            "eight source/workload files whose SHA-256 hashes are recorded in the environment. "
            "The generator validates those archived bytes before producing this report."
        ),
        f"- Shared workload SHA-256: {bt}{evidence['workload_hash']}{bt}.",
        (
            "- Later generic-runner validation fixes are not part of this full matrix. "
            "The matrix used JSON responses, zero retries, role/content-only messages, "
            "and an independent driver check rejecting pre-existing priming hits. "
            "Attribute these numbers to the archived snapshot, not to the latest source. "
            "A later smoke run is separate evidence and does not replace this 48-run matrix."
        ),
        (
            "- Final source smoke validation: "
            "[current-code completion](runs/final-smoke-20261110/completion.json) and "
            "[clean pinned-environment completion](runs/locked-env-smoke-20261110/completion.json) "
            "each record 8 runs, 32 measured requests, zero failures and four passed fault checks. "
            "Their raw archives are in [evidence/](evidence/); they are excluded from the matrix above."
        ),
        (
            "- Metadata correction: the archived driver originally labeled Redis as RESP2. "
            "After measurement, both installed synchronous and asynchronous redis-py "
            "connection defaults were inspected and found to use protocol 3. "
            "The environment now records RESP3 and an explicit correction note. "
            "Raw requests and measured source hashes were not changed by that correction."
        ),
        (
            f"- [Initial transport attempt]({raw}/initial_transport_attempt.json): "
            "a private Unix socket was unavailable and produced no matrix measurements. "
            "The reported matrix used an actual bundled Redis server over normal loopback TCP. "
            "The separate quick validation recorded there is not counted in the 48 runs."
        ),
        "",
        (
            "From the repository root, extract the raw-data archive if the request JSONL "
            "files are not already present, then regenerate without rerunning services:"
        ),
        "",
        FENCE + "bash",
        "mkdir -p benchmarks/runs",
        f"tar -xzf benchmarks/evidence/{run_dir.name}.tar.gz -C benchmarks/runs",
        ("python benchmarks/generate_report.py --run-dir benchmarks/runs/cpu-reference-20261110"),
        FENCE,
        "",
        (
            "To reproduce measurement, use the archived source files in a separate checkout "
            "and the recorded environment versions. Do not overwrite the preserved evidence. "
            "Running the current driver produces a new experiment with new provenance."
        ),
        "",
        "## Limits and work not run",
        "",
        (
            "- **Real model / GPU / vLLM benchmark: NOT RUN.** No model tokens/s, "
            "TTFT, TPOT, GPU utilization, precision comparison, or model speedup is reported."
        ),
        (
            "- **Held-out language-model quality and calibration: NOT RUN.** "
            "The deterministic sentinel checks protocol correctness only; they do not "
            "measure task accuracy, calibration, safety, reward quality, or semantic equivalence."
        ),
        (
            "- Nominal backend sleep is 20 ms; actual E2E includes other work and scheduling. "
            "No claim that observed latency should equal 20 ms follows."
        ),
        (
            "- Three short repetitions with 200 measured requests each have limited tail "
            "support. The bars show observed variation, not statistical confidence or "
            "independence across repeated requests."
        ),
        (
            "- Client, gateway, fixture, and Redis contend on the same shared host. "
            "Client loop-lag telemetry is preserved in run summaries but cannot isolate "
            "the causes of the observed latency or establish a remote-client result."
        ),
        (
            "- No traffic from real users, production uptime, sustained soak test, "
            "multi-node result, or production service-level guarantee is established."
        ),
        (
            "- A response-cache hit replays an existing answer. It is distinct from "
            "model warm-up, KV/prefix caching, and faster token generation."
        ),
        "",
        (
            "See [measurement protocol](PROTOCOL.md) and "
            "[interview practice notes](../docs/INTERVIEW_NOTES.md) for scope and next experiments."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/RESULTS.md")
    parser.add_argument(
        "--figure", type=Path, default=ROOT / "benchmarks/evidence/cpu-reference-20261110.png"
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report_path = args.report.resolve()
    figure_path = args.figure.resolve()
    evidence = load_evidence(run_dir)
    make_figure(evidence, figure_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        make_report(evidence, run_dir, report_path, figure_path), encoding="utf-8"
    )
    print(
        f"Validated {len(evidence['summaries'])} runs, {evidence['measured']} measured "
        f"and {evidence['priming']} priming requests; "
        f"matplotlib {matplotlib.__version__}"
    )
    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
