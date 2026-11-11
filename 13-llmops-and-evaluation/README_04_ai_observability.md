# Code Walkthrough — `04_ai_observability.ipynb`

> Module: **13-llmops-and-evaluation** | Notebook: **04_ai_observability.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 29
- **Markdown cells:** 24
- **Code cells:** 5

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

> # 04 — AI Observability
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
> ## 1. Concept Overview — AI observability — from traces to insights

> Observability is the practice of knowing what your system is doing *right
> now* — and being able to ask new questions about it without re-deploying.
> For LLM apps, observability combines traces (notebook 3) with metrics,
> logs, and dashboards.
> 
> **Beginner track:** learn the three pillars: metrics, logs, traces. They
> answer different questions.
> **Advanced track:** also learn drift detection, anomaly detection, and how
> to set up SLOs (service-level objectives) for an LLM service.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Three pillars of observability:
> 
> - **Metrics**: numeric time series — `requests_per_second`,
>   `latency_p95`, `error_rate`. Aggregatable.
> - **Logs**: discrete events with structured fields. Searchable.
> - **Traces**: tree of spans (notebook 3). Request-scoped.
> 
> SLOs: pick a metric (e.g. p95 latency), set a target (e.g. < 500ms),
> define a window (e.g. 28 days), and an error budget (e.g. 1% violations
> allowed).

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a tiny metrics aggregator: counter, gauge,
> and histogram. These are the three metric types every observability system
> supports.
> 
> **Why it matters:** Prometheus, DataDog, Honeycomb all expose these three
> primitives. Once you can implement them, you can use any observability
> backend.
> 
> **What to watch for:** a histogram is *not* a single value — it's a set of
> bucketed counts. P95 is computed by interpolating within buckets.


### Code cell #1
```python
import time
from collections import defaultdict

class Counter:
    """Monotonically increasing value (e.g. total requests)."""
    def __init__(self, name):
        self.name = name; self.value = 0
    def inc(self, n=1): self.value += n
    def get(self): return self.value

class Gauge:
    """Value that can go up or down (e.g. active connections)."""
    def __init__(self, name):
        self.name = name; self.value = 0
    def set(self, v): self.value = v
    def inc(self, n=1): self.value += n
    def dec(self, n=1): self.value -= n
    def get(self): return self.value

class Histogram:
    """Distribution of values in buckets (e.g. request latency)."""
    def __init__(self, name, buckets=None):
        self.name = name
        self.buckets = buckets or [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        self.counts = [0] * (len(self.buckets) + 1)
        self.sum = 0.0
    def observe(self, v):
        self.sum += v
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1; return
        self.counts[-1] += 1
    def quantile(self, q):
        # Linear interpolation within a bucket.
        total = sum(self.counts)
        target = q * total
        cum = 0
        for i, c in enumerate(self.counts):
            cum += c
            if cum >= target:
                if i == 0: return 0
                if i == len(self.buckets): return self.buckets[-1]
                lo = self.buckets[i-1]; hi = self.buckets[i]
                return lo + (hi - lo) * (target - (cum - c)) / c
        return self.buckets[-1]

# Demo.
req_count = Counter("requests_total")
active = Gauge("active_requests")
latency = Histogram("request_latency_seconds")

for _ in range(100):
    req_count.inc()
    active.inc(); time.sleep(0.001); active.dec()
    latency.observe(0.05 + 0.001 * (_ % 10))

print(f"total requests: {req_count.get()}")
print(f"current active: {active.get()}")
print(f"latency p50: {latency.quantile(0.5)*1000:.1f} ms")
print(f"latency p95: {latency.quantile(0.95)*1000:.1f} ms")
print(f"latency p99: {latency.quantile(0.99)*1000:.1f} ms")
```

**What this code does:**

- **Headline:** `import time`
- **Defines:** class `Counter` — Monotonically increasing value (e.g. total requests).; class `Gauge` — Value that can go up or down (e.g. active connections).; class `Histogram` — Distribution of values in buckets (e.g. request latency).
- **Key imports:** `import time`, `from collections import defaultdict`

> **Takeaway:** Observability primitives are simple — counters, gauges,
> histograms. The complexity is in *collecting* them at scale and *querying*
> them efficiently. That's what backends (Prometheus, Datadog) do.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses the `prometheus_client` library — the standard
> Python library for emitting Prometheus metrics. If it's not installed, we
> skip.
> 
> **Why it matters:** Prometheus is the de-facto open-source monitoring
> stack. Knowing the Python client means you can instrument any Python LLM
> service for free.
> 
> **What to watch for:** `prometheus_client.start_http_server(8000)` exposes
> metrics at `http://localhost:8000/metrics`. A Prometheus scraper polls
> that endpoint at intervals.


### Code cell #2
```python
try:
    from prometheus_client import Counter as PromCounter, Histogram as PromHist, start_http_server
    requests = PromCounter("llm_requests_total", "Total LLM requests", ["model"])
    latency = PromHist("llm_request_seconds", "LLM request latency",
                       buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
    # Simulate some observations.
    requests.labels(model="gpt-4o-mini").inc()
    requests.labels(model="gpt-4o-mini").inc()
    latency.observe(0.42)
    print("prometheus_client installed; metrics would be scraped at /metrics")
    print("in production: start_http_server(8000) and a prometheus scraper polls it")
except ImportError:
    print("prometheus_client not installed — pip install prometheus_client")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from prometheus_client import Counter as PromCounter`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Generates a mock time series of LLM request latencies
> and labels: normal traffic, a latency spike (degraded model), and a cost
> spike (much longer outputs). We'll use this for anomaly detection.
> 
> **Why it matters:** time-series data is the raw material of observability.
> Without it, you can't detect drift, spikes, or anomalies.
> 
> **What to watch for:** real LLM services have *periodic* patterns
> (weekday vs weekend, hour-of-day). Our mock data ignores that; real
> anomaly detection must account for it.


### Code cell #3
```python
import numpy as np
rng = np.random.default_rng(0)
n = 200  # 200 minutes of traffic

# Baseline: ~200ms latency, ~100 tokens per request.
latencies = rng.normal(0.20, 0.03, n)
tokens = rng.normal(100, 15, n)

# Inject a latency spike at minute 100-110.
latencies[100:110] += 0.5
# Inject a token-count spike at minute 150-160 (longer outputs).
tokens[150:160] *= 3

time_series = {"minute": list(range(n)),
              "latency_s": latencies.round(3).tolist(),
              "tokens": tokens.round(0).astype(int).tolist()}
print(f"time series length: {n} minutes")
print(f"latency stats: mean={latencies.mean():.3f}s, max={latencies.max():.3f}s")
print(f"token stats:   mean={tokens.mean():.0f}, max={tokens.max():.0f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the latency and token-count time series with
> the injected anomalies highlighted. This is the standard "ops dashboard"
> view.
> 
> **Why it matters:** visualising time series is the first step of any
> observability investigation. Spikes are obvious on a chart long before
> they show up in an alert.
> 
> **What to watch for:** we plot raw values. Real dashboards also show
> moving averages, anomalies, and SLO thresholds as horizontal lines.


### Code cell #4
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

axes[0].plot(time_series["minute"], time_series["latency_s"], color="steelblue")
axes[0].axvspan(100, 110, color="red", alpha=0.2, label="latency spike")
axes[0].set_ylabel("latency (s)"); axes[0].legend()
axes[0].set_title("LLM request latency over time")

axes[1].plot(time_series["minute"], time_series["tokens"], color="orange")
axes[1].axvspan(150, 160, color="red", alpha=0.2, label="token spike")
axes[1].set_ylabel("tokens per request"); axes[1].set_xlabel("minute")
axes[1].legend()
axes[1].set_title("LLM output token count over time")

plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Observability is evaluated by *coverage* and *cost*:
> 
> - **Coverage**: what fraction of requests have full traces? Aim for 100% on critical paths.
> - **Sampling rate**: for high-traffic services, you may sample 1-10% of traces.
> - **Cost**: storing every trace is expensive. Sample by error / slow / expensive.
> - **SLO adherence**: in the last 28 days, did the service meet its SLO?

> **What this does:** A simple anomaly detector: flag any point that's more
> than 3 standard deviations from the rolling mean. This is the simplest
> anomaly detection algorithm.
> 
> **Why it matters:** anomaly detection is the value-add of observability.
> Without it, you have to stare at dashboards; with it, the system pages you
> when something goes wrong.
> 
> **What to watch for:** 3-sigma works for Gaussian data. Real traffic has
> long tails — use the IQR rule or a non-parametric detector.


### Code cell #5
```python
def detect_anomalies(values, window=20, threshold=3.0):
    """Flag points > threshold std devs from the rolling mean."""
    flags = [False] * len(values)
    for i in range(window, len(values)):
        history = values[i-window:i]
        mean = sum(history) / window
        std = (sum((v - mean) ** 2 for v in history) / window) ** 0.5
        if std > 0 and abs(values[i] - mean) > threshold * std:
            flags[i] = True
    return flags

latency_anomalies = detect_anomalies(time_series["latency_s"])
token_anomalies = detect_anomalies(time_series["tokens"])
print(f"latency anomalies: {sum(latency_anomalies)} flagged at minutes: {[i for i, f in enumerate(latency_anomalies) if f][:5]}...")
print(f"token anomalies  : {sum(token_anomalies)} flagged at minutes: {[i for i, f in enumerate(token_anomalies) if f][:5]}...")
```

**What this code does:**

- **Headline:** `def detect_anomalies(values, window=20, threshold=3.0):`
- **Defines:** def `detect_anomalies` — Flag points > threshold std devs from the rolling mean.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace 3-sigma with the IQR rule — does it catch the same anomalies?
> 2. Add a seasonal component (weekday vs weekend) and detect anomalies against the seasonal mean.
> 3. Set an SLO: p95 latency < 500ms. Compute the error budget burned over the 200 minutes.
> 4. Use a real observability backend (Langfuse, LangSmith, or Prometheus + Grafana).

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a tail-based sampler that keeps 100% of error traces, 1% of normal.
> 2. Build a drift detector that compares this week's token distribution to last week's.
> 3. Implement an SLO burn-rate alert that pages when the error budget is being spent too fast.
> 4. Build a dashboard that combines metrics + logs + traces for a single request.
> 5. Implement cost attribution: which user/team/account generated this trace's cost?

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What are the three pillars of observability?
> 2. What is the difference between a counter and a gauge?
> 3. How is a histogram different from a gauge?
> 4. What is an SLO, and what is an error budget?
> 5. Why sample traces? When would you not sample?
> 6. What is tail-based sampling, and why is it useful for LLM apps?
> 7. How would you detect drift in an LLM service?
> 8. What is the difference between Prometheus and Datadog?
