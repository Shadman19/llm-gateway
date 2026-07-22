# START HERE

Follow these in order. Stop at any step that fails and report the error.

---

## Step 1 — Open Terminal in this folder

Unzip this file, then open the Terminal app and type `cd ` (with a space),
then drag the `llm-gateway` folder onto the Terminal window and press Enter.

Check you are in the right place:

```bash
ls
```

You should see: `app`, `tests`, `benchmarks`, `Dockerfile`, `README.md`, and others.

---

## Step 2 — Run the tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

**Expected result:** `14 passed`

If you see 14 passed, the code works. Nothing is broken.

---

## Step 3 — Put it on GitHub

The repository must be named exactly `llm-gateway`, because the CI badge in
the README points to that name.

```bash
git init
git add -A
git commit -m "LLM serving gateway with tests and CI"
```

Then create an empty repo named `llm-gateway` at github.com/new
(public, do not add a README), and run the two commands GitHub shows you.

After pushing, open the **Actions** tab on GitHub. A workflow will run.
Wait for the green checkmark.

**At this point you already have a real artifact:** a tested, CI-verified repo.

---

## Step 4 — Rent a GPU (about 3 hours, roughly $2)

Go to runpod.io or vast.ai. Choose an A10 or A6000 GPU pod with a PyTorch image.

On the GPU machine:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8001
```

Wait until it prints that the server is running. Copy the pod's public IP.

---

## Step 5 — Connect your gateway to the model

Back on your own Mac, in the llm-gateway folder:

```bash
export VLLM_URL=http://PASTE_POD_IP_HERE:8001/v1
docker compose up --build
```

In a second Terminal window:

```bash
curl localhost:8000/health
```

You want to see `"backend_reachable": true`.

Then send a real request:

```bash
curl -s localhost:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain KV caching in two sentences."}],"max_tokens":128}'
```

---

## Step 6 — Measure

On the GPU machine:

```bash
vllm bench serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --base-url http://localhost:8001 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 256 --random-output-len 128 \
  --max-concurrency 16
```

Run it again with `--max-concurrency 1`, `8`, and `32`.

On your Mac:

```bash
locust -f benchmarks/locustfile.py --host http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 2m --headless
```

Run it again with `--users 1` and `--users 50`.

---

## Step 7 — Fill in the numbers

Open `benchmarks/RESULTS.md`. Replace every `TBD` with a number you actually
measured. Fill the environment table at the top first (GPU model, memory,
vLLM version).

Copy the headline numbers into the table near the top of `README.md`.

Commit and push:

```bash
git add -A
git commit -m "Add measured benchmark results"
git push
```

**Done.** Never write a number you did not measure.
