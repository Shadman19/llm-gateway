"""Gateway load test.

Measures the full request path: validation, cache, routing, backend, response.
Run the model server and gateway first, then:

    locust -f benchmarks/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 3m --headless --csv=benchmarks/run

Three request classes are tagged separately so warm-cache and cold-cache
latency can be reported as distinct numbers rather than blended into one mean.
"""
import random
import uuid

from locust import HttpUser, between, task

WARM_PROMPT = "Summarize the causes of flight departure delays in two sentences."


class GatewayUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def warm_cache(self):
        """Identical prompt every time. After the first request this is a cache hit."""
        self.client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": WARM_PROMPT}], "max_tokens": 128},
            name="POST /v1/chat [warm cache]",
        )

    @task(2)
    def cold_cache(self):
        """Unique prompt every time. Always reaches the model."""
        prompt = f"Explain concept {uuid.uuid4()} in one paragraph."
        self.client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 128},
            name="POST /v1/chat [cold cache]",
        )

    @task(1)
    def cache_bypassed(self):
        """Isolates gateway overhead against a repeated backend call."""
        self.client.post(
            "/v1/chat",
            json={
                "messages": [{"role": "user", "content": WARM_PROMPT}],
                "max_tokens": 128,
                "use_cache": False,
            },
            name="POST /v1/chat [no cache]",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
