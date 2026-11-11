# Code Walkthrough — `04_model_serving.ipynb`

> Module: **14-ai-infrastructure** | Notebook: **04_model_serving.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 31
- **Markdown cells:** 25
- **Code cells:** 6

## 10-point learning structure

Every notebook in this cookbook follows this structure wherever applicable:

1. **Concept Overview** — the problem and its fundamentals
2. **Mathematical Foundation** — the underlying mathematics
3. **Implementation from Scratch** — pure Python + NumPy
4. **Library Implementation** — the industry-standard framework
5. **Dataset Exploration** — a real dataset
6. **Visualization** — results through plots
7. **Model Evaluation** — appropriate metrics
8. **Experiments** — modify and observe
9. **Hands-on Challenges** — practice independently
10. **Interview Questions** — reinforce understanding

---

> # 04 — Model Serving
> 
> > **ML & AI Engineering Cookbook** — hands-on Jupyter notebook.
> >
> > Run on [Google Colab](https://colab.research.google.com/) or locally with Jupyter.

> ## Notebook Learning Structure
> 
> 1. **Concept Overview** — understand the problem and its fundamentals.
> 2. **Mathematical Foundation** — explore the underlying mathematics.
> 3. **Implementation from Scratch** — build it with Python and NumPy.
> 4. **Library Implementation** — use the industry-standard framework.
> 5. **Dataset Exploration** — work with a practical dataset.
> 6. **Visualization** — understand results through plots.
> 7. **Model Evaluation** — analyze performance with metrics.
> 8. **Experiments** — modify parameters and observe behavior.
> 9. **Hands-on Challenges** — practice independently.
> 10. **Interview Questions** — reinforce conceptual understanding.

> ---
> 
> ## 1. Concept Overview — Model serving — from notebook to production

> A model in a notebook is a prototype. A *served* model is one that's
> available over HTTP, gRPC, or a streaming endpoint. This notebook covers
> the standard Python serving stack: FastAPI, Pydantic, streaming, and
> deployment to a server.
> 
> **Beginner track:** learn FastAPI basics: a single `POST /predict`
> endpoint with Pydantic validation. That's enough for a v0.
> **Advanced track:** also learn streaming responses, async, and deployment
> with Docker + Uvicorn + Gunicorn.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A model server is a function $f: \text{HTTP request} \to \text{HTTP response}$
> that wraps a model. Request validation is via Pydantic schemas; response
> is JSON (or SSE for streaming). Concurrency is via `asyncio`; CPU-bound
> work is offloaded to a thread pool.
> 
> Throughput = $\min(\text{model throughput}, \text{server throughput})$.
> Adding async only helps if the model is non-blocking.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a minimal HTTP server using only the standard
> library (`http.server`). 30 lines. No framework, no validation, no async.
> 
> **Why it matters:** FastAPI hides a lot. Once you've written a server in
> pure Python, the FastAPI abstractions (routing, validation, OpenAPI) make
> sense.
> 
> **What to watch for:** `BaseHTTPRequestHandler` is single-threaded by
> default. For concurrency, use `ThreadingHTTPServer` or a real framework.


### Code cell #1
```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Mock model: returns the length of the input.
def predict(text):
    return {"length": len(text), "words": len(text.split())}

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
            text = req.get("text", "")
            result = predict(text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_error(400, str(e))

    def log_message(self, *args): pass  # quiet

# Demo (in a real notebook you would not start a server; we just show the code).
print("Pattern: server = HTTPServer(('localhost', 8000), Handler); server.serve_forever()")
print("Demo: predict('hello world') =", predict("hello world"))
```

**What this code does:**

- **Headline:** `from http.server import BaseHTTPRequestHandler, HTTPServer`
- **Defines:** def `predict`; class `Handler`
- **Key imports:** `from http.server import BaseHTTPRequestHandler`, `import json`

> **Takeaway:** HTTP serving = read request → call function → write response.
> FastAPI, Flask, Django all do this with extra structure. The core is trivial.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds the same server with FastAPI + Pydantic. The
> code is half as long and you get validation, OpenAPI docs, and async for
> free.
> 
> **Why it matters:** FastAPI is the standard Python serving stack for ML
> in 2024-2025. Every LLM service I've seen in production uses it.
> 
> **What to watch for:** `uvicorn.run(app, host, port)` starts the server.
> In a notebook we just show the code; running `uvicorn.run` blocks the
> cell.


### Code cell #2
```python
try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    app = FastAPI(title="Mini LLM Server", version="0.1.0")

    class PredictRequest(BaseModel):
        text: str = Field(..., description="Input text to process.")
        max_tokens: int = Field(100, ge=1, le=1000, description="Max output tokens.")

    class PredictResponse(BaseModel):
        text: str
        tokens: int

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(req: PredictRequest):
        # In production: call your LLM here.
        return PredictResponse(text=f"Echo: {req.text}", tokens=len(req.text.split()))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    print("FastAPI app defined. To serve:")
    print("  uvicorn <module>:app --host 0.0.0.0 --port 8000")
    print("OpenAPI docs at http://localhost:8000/docs")
except ImportError as e:
    print(f"fastapi not installed ({e}) — pip install fastapi uvicorn")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from fastapi import FastAPI`, `from pydantic import BaseModel`

> **What this does:** Demonstrates a streaming endpoint — essential for LLM
> serving where you want to return tokens as they're generated, not all at
> once. Uses FastAPI's `StreamingResponse`.
> 
> **Why it matters:** users perceive streaming responses as 5-10x faster than
> batch responses, even if total time is identical. Every ChatGPT-class UI
> uses streaming.
> 
> **What to watch for:** streaming uses Server-Sent Events (SSE) over HTTP.
> Each chunk is a `data: <json>\n\n` line. The client buffers chunks and
> renders them as they arrive.


### Code cell #3
```python
try:
    from fastapi.responses import StreamingResponse
    import asyncio, json

    async def stream_tokens(text, n=10):
        """Mock streaming generator: yield tokens one at a time."""
        for i in range(n):
            await asyncio.sleep(0.1)  # simulate model latency
            yield f"data: {json.dumps({'token': f'token_{i}', 'index': i})}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/stream")
    async def stream_endpoint(req: PredictRequest):
        return StreamingResponse(stream_tokens(req.text, req.max_tokens),
                                 media_type="text/event-stream")

    print("Streaming endpoint added at /stream")
    print("Client opens an SSE connection; server yields tokens as they're generated.")
except ImportError:
    print("fastapi not installed")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from fastapi.responses import StreamingResponse`, `import asyncio`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a mock serving load profile: 100 requests with
> varying prompt lengths. We'll use it for capacity planning.
> 
> **Why it matters:** capacity planning is the *operational* question — how
> much hardware do I need for my traffic? You need a representative load
> profile.
> 
> **What to watch for:** real traffic has *bursts*. Average load is
> useless for capacity planning; you need to size for the 99th percentile.


### Code cell #4
```python
import numpy as np
rng = np.random.default_rng(0)
n = 100
prompt_lens = rng.integers(10, 500, n)
output_lens = rng.integers(10, 200, n)
inter_arrivals = rng.exponential(scale=0.1, size=n)  # seconds between requests

workload = [{"prompt_len": int(p), "output_len": int(o), "wait_s": float(w)}
            for p, o, w in zip(prompt_lens, output_lens, inter_arrivals)]
print(f"workload size: {len(workload)} requests")
print(f"avg prompt length: {prompt_lens.mean():.0f} tokens")
print(f"avg output length: {output_lens.mean():.0f} tokens")
print(f"avg inter-arrival: {inter_arrivals.mean():.3f} s  -> {1/inter_arrivals.mean():.1f} req/s")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the request arrival pattern over time. Bursty
> traffic is the norm; you need to size for the peaks, not the mean.
> 
> **Why it matters:** this is the chart you'd show a stakeholder to explain
> why you need 3x the average capacity. Without it, "1 req/s average" sounds
> like 1 server; in reality it needs 3-5.
> 
> **What to watch for:** the y-axis is requests per second, smoothed over a
> rolling 1-second window. Real load tests use this exact chart.


### Code cell #5
```python
import matplotlib.pyplot as plt

arrivals = np.cumsum(inter_arrivals)

fig, ax = plt.subplots(figsize=(10, 4))
ax.step(arrivals, np.arange(1, n+1), where="post", color="steelblue")
ax.set_xlabel("time (s)"); ax.set_ylabel("cumulative requests")
ax.set_title("Request arrival pattern (exponential inter-arrivals)")
plt.tight_layout(); plt.show()
print(f"peak rate: ~{1/inter_arrivals.min():.0f} req/s")
print(f"mean rate: ~{1/inter_arrivals.mean():.0f} req/s")
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Serving systems are evaluated by:
> 
> - **Throughput** — requests/sec the server can sustain.
> - **Latency** — p50, p95, p99 of response time.
> - **Error rate** — fraction of requests that fail (5xx).
> - **Concurrent users** — how many simultaneous connections can it handle?

> **What this does:** A simple load-test driver: send N requests with the
> inter-arrival times from the workload, measure response latency. We mock
> the server (no real model) to show the pattern.
> 
> **Why it matters:** a load-test script is the *minimum* tool for serving
> eval. Without it, you don't know your service's limits.
> 
> **What to watch for:** real load tests use `locust`, `wrk`, or `k6` — they
> have features for distributed load generation and detailed reporting.


### Code cell #6
```python
import time

# Mock load test: simulate requests with the workload profile.
mock_latencies = []
for i, req in enumerate(workload[:30]):
    # Mock latency: depends on output length (token generation time).
    latency = req["output_len"] * 0.02 + rng.uniform(0.01, 0.05)  # 20ms per token + noise
    mock_latencies.append(latency)
    time.sleep(0)  # don't actually sleep in the demo

import numpy as np
latencies = np.array(mock_latencies)
print(f"latency p50: {np.percentile(latencies, 50)*1000:.1f} ms")
print(f"latency p95: {np.percentile(latencies, 95)*1000:.1f} ms")
print(f"latency p99: {np.percentile(latencies, 99)*1000:.1f} ms")
print(f"throughput (mock): {1 / latencies.mean():.1f} req/s")
```

**What this code does:**

- **Headline:** `import time`
- **Key imports:** `import time`, `import numpy as np`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a `__call__` method to your model class so it's callable from FastAPI.
> 2. Use `asyncio.gather` to handle multiple requests concurrently.
> 3. Deploy the FastAPI app with `uvicorn` and `gunicorn` workers.
> 4. Containerize the server with Docker and run it locally.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Build a `/chat` endpoint that maintains conversation history per session.
> 2. Implement rate limiting (e.g. 10 req/min per IP).
> 3. Build an OpenTelemetry-instrumented FastAPI server.
> 4. Implement graceful shutdown: drain in-flight requests before exit.
> 5. Build a load-testing harness with `locust` that simulates 1000 concurrent users.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between FastAPI and Flask?
> 2. Why use Pydantic for request validation?
> 3. What is SSE, and why is it preferred for LLM serving?
> 4. How would you handle concurrent requests in a model server?
> 5. What is the difference between sync and async endpoints in FastAPI?
> 6. How do you load-test an LLM server?
> 7. What is graceful shutdown, and why does it matter?
> 8. How would you deploy a FastAPI server to production?
