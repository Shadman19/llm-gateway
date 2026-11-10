# Interview practice: measured gateway reliability

These notes describe a local engineering experiment. They are not credentials,
production experience, a model-quality study, or evidence of GPU performance.
Use the [results](../benchmarks/RESULTS.md) and raw evidence when discussing numbers.

## 1. What did the warm-cache result actually measure?

**Answer:** Exact response replay from Redis through the gateway. The CPU fixture
generated deterministic text, with a nominal 20 ms asynchronous delay. A warm run
primed 200 unique payloads, then measured the same 200 requests. The measured
responses had to report **cached=true** and match their primed content. Those
priming requests were excluded from the measured throughput and latency.

This differs from GPU/model warm-up and from an upstream KV or prefix cache.
The fixture had no model, tokenizer, or GPU. Lower response-replay latency is
not faster token generation, and the non-streaming gateway cannot observe TTFT.

**Practice:** Explain the four paths in one minute. Identify which backend
calls occur during priming and which occur during measurement. Then explain why
an experiment with a real vLLM backend must be separately labeled.

## 2. How do you prevent a cache hit for the wrong request?

**Answer:** The measured cache key hashes a canonical description containing the
backend name and endpoint, model, declared revision and dtype, ordered messages,
output budget, temperature, seed, and namespace. Changing a relevant setting
must change the key. Distinct namespaces separate runs without flushing another
workload's keys. The recorded experiment used a 3,600-second cache TTL and fresh
namespaces, then checked actual hit/miss outcomes.

Revision and dtype strings are operator declarations; the gateway does not
attest that a remote server loaded those exact weights. A namespace is also not
an authorization boundary. This project does not establish secure multi-tenant
serving.

**Practice:** Trace the key in the archived
[app/cache.py](../benchmarks/runs/cpu-reference-20261110/source_snapshot/app/cache.py).
Name three changes that require cache separation. Explain why identical prompt
text alone is insufficient, and why a TTL does not solve model-version identity.

## 3. How did you make the comparison fair, and what do the bars mean?

**Answer:** The four paths used the same 200 fixed payloads and generation
settings at each concurrency. Cold means response-cache cold, not GPU cold.
The experiment ran each path separately and rotated the order within each
concurrency/repetition block. It collected three repetitions at concurrency
1, 4, 8, and 16: 48 runs and 9,600 measured requests, plus 2,400 warm-prime
requests.

Each plotted point is the median of three run summaries. The bars show their
observed minimum and maximum, not a confidence interval. The request records
share one host and repeated workload; they are not independent observations
of user traffic. Subtracting unrelated P50 summaries does not isolate gateway
overhead. Use matched per-request analysis and additional instrumentation for
that separate question.

**Practice:** Rebuild the [figure](../benchmarks/evidence/cpu-reference-20261110.png)
with the command below. Locate the widest warm-cache P95 range and explain what
can and cannot be concluded from three runs.

    python benchmarks/generate_report.py

## 4. What does zero failed measured requests say about failure handling?

**Answer:** Only that the normal-workload matrix recorded zero failures in
9,600 measured requests under these conditions. Four controlled checks were
reported separately: non-JSON upstream output returned HTTP 502; an upstream
timeout returned 503; unavailable-backend readiness returned 503 while liveness
remained 200; and stopped Redis still allowed an uncached HTTP 200 response.

The first three recorded fault intervals include a following liveness request,
so they are not isolated timeout measurements. Separate cache-read, cache-write,
and backend timeout budgets are not one overall request deadline. A live
process is not necessarily ready to serve a model. Schema validation must reject
malformed output rather than caching it as a valid answer.

**Practice:** Read [fault_smoke.json](../benchmarks/runs/cpu-reference-20261110/fault_smoke.json).
Explain why expected 502/503 responses were excluded from the healthy-load
failure count. Sketch a test that distinguishes liveness, readiness, and
end-to-end request-deadline behavior without making a production SLO claim.

## 5. How would you defend reproducibility and state the limitations?

**Answer:** Start with the measured
[source_snapshot/](../benchmarks/runs/cpu-reference-20261110/source_snapshot/)
and the SHA-256 hashes in
[environment.json](../benchmarks/runs/cpu-reference-20261110/environment.json).
The recorded upstream commit is a base; it does not by itself identify the
working-tree changes used in the experiment. Later runner fixes and their short
smoke validation are separate evidence, not a replacement for this full matrix.
The report generator checks archived source bytes, summary/aggregate agreement,
raw counts, cache outcomes, and deterministic content hashes.

The Redis protocol label was corrected after measurement from RESP2 to RESP3
after inspecting both installed redis-py connection defaults. The correction is
explicit in metadata; measured requests and source hashes were retained.
The experiment used real Redis over loopback TCP after the initial Unix-socket
attempt was unavailable and produced no matrix results.

All services and the client shared a CPU host. Scheduling, HTTP overhead, and
load-generator contention affect E2E latency; a nominal 20 ms fixture sleep is
not the observed response latency. No real LLM, held-out language-model quality,
calibration, quantization comparison, sustained user load, multi-node system,
or production operating history was measured.

**Practice:** Give a 90-second project explanation with one measured finding,
one implementation decision, and two limitations. Propose one next experiment:
a separately versioned real-model run on existing hardware, with a held-out
paired quality protocol before claiming any quality-preserving optimization.
