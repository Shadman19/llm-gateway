# Gateway measurement protocol

The unit of comparison is the same prompt and generation configuration sent
through a declared serving path. A response-cache hit reuses a prior answer; it
does not demonstrate faster model inference.

## Paths

- `direct-json`: complete JSON response from the upstream API.
- `gateway-bypass`: the same payload through `/v1/chat`, `use_cache=false`.
- `gateway-cold`: unique fixed payloads with a fresh, isolated response-cache
  namespace. Every measured response must report `cached=false`.
- `gateway-warm`: prime the fixed payloads in a fresh namespace, then measure
  exact replay. Every measured response must report `cached=true` and equal its
  primed answer byte for byte. Priming is excluded from the measured phase.
- `direct-stream`: a separate streaming experiment. TTFT starts at dispatch and
  ends at the first nonempty content chunk, excluding empty protocol events.

The non-streaming gateway reports no TTFT. Response-cache coldness says nothing
about GPU startup, model warmup, or the upstream KV/prefix cache.

## Workload and provenance

Keep the prompt JSONL, order, output-token budget, model, temperature and seed
fixed across comparable paths. Keep client configuration and concurrency fixed.
The runner records a workload hash, every attempt, status/schema failures,
retries, queue delay and full logical-request latency. Health checks run outside
the measured workload. Cold runs require distinct request payloads rather than
changing prompt text between paths. Never flush another workload's cache.

Record Python and dependency versions, the source base and working-tree hash,
actual hardware, server configuration, client placement, model revision/dtype,
prefix-cache configuration and raw output locations. Revision and dtype labels
are operator declarations, not backend attestations.

Use at least three repetitions at each concurrency setting. Report per-run
sample counts, failures, P50/P95 and replicate variation. P99 from 200 requests
has little tail support and does not establish a service-level guarantee.
Compare paired requests from matched workloads; subtracting unrelated percentile
summaries is not an estimate of gateway overhead.

## Controlled failures

Keep expected failures separate from successful throughput measurements. Check
unavailable Redis, upstream timeout, non-JSON/malformed upstream responses,
corrupt cache entries, and isolation across backend endpoints and generation
settings. Cache reads and writes each have their own I/O budget; their sum is
not a single overall request deadline. The process can be live while its model
backend is unavailable: `/health` measures liveness, `/ready` checks readiness.

## Scope of the CPU reference experiment

The local reference stack uses an actual Redis process and a deterministic HTTP
fixture with nominal 20 ms delay. It checks middleware behavior and the
benchmark protocol. It does not run an LLM, tokenizer, GPU, vLLM engine, model
compression or quantization. No tokens/second, model TTFT, production SLO,
language-model quality, or inference acceleration follows from these results.
All services and the client share one host; scheduling and load-generator
contention remain limitations.

For a real model, run a separately labeled experiment against the configured
server. Pin its revision and record its actual GPU/vLLM configuration. Before
comparing precision or quantization settings, define a separate held-out paired
quality evaluation and a task-appropriate noninferiority criterion.

## Focused study and next experiments

- Use the official [vLLM serving benchmark reference](https://docs.vllm.ai/en/latest/cli/bench/serve/)
  to learn model-server metrics and run a real backend when suitable existing
  hardware is available. Its random-prompt benchmark is a separate workload.
- Use the [OpenTelemetry Python example](https://opentelemetry.io/docs/languages/python/getting-started/)
  for a later tracing exercise: distinguish gateway, cache and backend spans,
  without putting prompt text or credentials in telemetry.
- The [paired binary-quality utility](QUALITY_GATE.md) now validates externally
  supplied score pairs, preserves input hashes, and reports conservative uncertainty.
  Its tests are synthetic; actual held-out model quality remains unmeasured.
- Collect a separately held-out quality regression suite before advertising any
  precision, quantization or caching policy as preserving model quality.

These are study tasks and proposed extensions, not completed credentials or
published research. A paper would require a novel, supported finding beyond
the reference-stack checks.
