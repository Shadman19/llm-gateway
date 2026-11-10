# Start here

## 1. Install a reproducible local environment

Use Python 3.12. The pinned CPU reference requirements were checked on Linux.
They include an embedded Redis executable; no model download, GPU or paid API is
needed for the reference experiment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-benchmark.txt
python -m pytest
```

The tests exercise routing, backend contracts, cache isolation, dependency
outages, deadlines and benchmark validity. Their scope is recorded in the test
names and coverage output.

## 2. Run the CPU reference experiment

```bash
python -m benchmarks.reference_stack \
  --output-dir benchmarks/runs/my-cpu-reference
```

The driver creates an isolated Redis process, a deterministic HTTP fixture and
the gateway. It measures the same fixed workload through four paths, at
concurrency 1/4/8/16 with three repetitions. It then runs controlled failure
checks, writes raw records/metadata and stops the services. Choose a new output
directory for each experiment; prior artifacts are preserved.

A smaller integration check is:

```bash
python -m benchmarks.reference_stack \
  --output-dir /tmp/gateway-reference-smoke \
  --requests 4 --concurrency 1,4 --replicates 1
```

Read `completion.json`, `aggregate.json`, `fault_smoke.json` and the individual
summary/JSONL files. Verify each run's validity, failures and sample counts.
The fixture is a model-shaped deterministic HTTP service. Its timings support
middleware and harness conclusions only.

## 3. Connect an existing real model server

Use existing authorized hardware or an approved endpoint. Keep a model server
on a private network or an appropriate tunnel. A local vLLM example is:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8001
```

With a separately running local Redis, start the gateway on the same machine:

```bash
export VLLM_URL=http://127.0.0.1:8001/v1
export VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export REDIS_URL=redis://127.0.0.1:6379/0
export MODEL_REVISION=the-actual-model-revision
export INFERENCE_DTYPE=the-actual-dtype
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

These labels must describe the actual deployment. `/health` is process liveness;
check `/ready` for the configured model's readiness. Docker Compose requires a
private model endpoint reachable from its container network; a host loopback
address inside a container refers to that container.

## 4. Measure a real backend with matched inputs

Run each path independently with the same prompt JSONL, concurrency and
sampling settings. Use a fresh run ID and output prefix each time. For example:

```bash
python -m benchmarks.serving_benchmark \
  --url http://127.0.0.1:8001/v1/chat/completions \
  --mode direct-json --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts benchmarks/fixtures/prompts.jsonl \
  --requests 200 --concurrency 8 --run-id model-direct-r1 \
  --output-prefix benchmarks/runs/model-direct-r1

python -m benchmarks.serving_benchmark \
  --url http://127.0.0.1:8000/v1/chat \
  --mode gateway-bypass --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts benchmarks/fixtures/prompts.jsonl \
  --requests 200 --concurrency 8 --run-id model-bypass-r1 \
  --output-prefix benchmarks/runs/model-bypass-r1
```

The bundled prompts are harness fixtures. Use a documented, task-appropriate
corpus for model capability or quality claims. Consult the
[protocol](benchmarks/PROTOCOL.md) before adding cold/warm response-cache,
streaming or precision comparisons. Non-streaming gateway TTFT remains unknown.

## 5. Write only supported conclusions

Record the actual hardware, model revision, precision, server versions,
configuration, client placement, raw request records and repeated-run results.
Keep real-model results distinct from the CPU reference stack. A speed or
quality claim needs its own completed experiment; placeholders are not results.

