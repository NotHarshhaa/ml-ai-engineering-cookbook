# Code Walkthrough — `03_tracing_observability.ipynb`

> Module: **13-llmops-and-evaluation** | Notebook: **03_tracing_observability.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 33
- **Markdown cells:** 26
- **Code cells:** 7

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

> # 03 — Tracing & Observability
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
> ## 1. Concept Overview — LLM tracing and observability

> An LLM app makes many calls: prompts, tool calls, retrieval, sub-agent
> calls. *Tracing* records each call's inputs, outputs, latency, and cost
> in a single tree. *Observability* is the practice of monitoring those
> traces in production.
> 
> **Beginner track:** learn the trace tree: spans, parent-child
> relationships, attributes. That's the core abstraction.
> **Advanced track:** also learn OpenTelemetry, distributed tracing across
> microservices, and tools like LangSmith, Langfuse, Arize Phoenix.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A trace is a tree of *spans*. Each span has: name, start/end time,
> attributes (key-value), parent span ID, status. The trace ID ties related
> spans together. Total latency = root span duration. Cost = sum of span
> costs (token count × price).
> 
> OpenTelemetry is the standard format: spans are exported as OTLP, stored
> in a backend (Jaeger, Tempo, LangSmith), queried for analysis.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a tiny tracing system in pure Python: a
> `Span` class with start/end timestamps, attributes, and a parent ID.
> We build a tree of spans, then "render" it as a text tree.
> 
> **Why it matters:** every tracing library (OpenTelemetry, LangSmith,
> Langfuse) implements this same abstraction. The complexity is in the
> backends and UIs, not the span itself.
> 
> **What to watch for:** spans are *start-end pairs*, not just timestamps.
> If you only record end times, you can't compute self-time (duration minus
> child duration).


### Code cell #1
```python
import time, uuid
from typing import Optional

class Span:
    """A single span in a trace: name, start/end, attributes, parent."""
    def __init__(self, name, parent_id=None, trace_id=None):
        self.span_id = uuid.uuid4().hex[:8]
        self.trace_id = trace_id or uuid.uuid4().hex[:8]
        self.parent_id = parent_id
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time = None
        self.attributes = {}
        self.children = []

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def end(self):
        self.end_time = time.perf_counter()

    @property
    def duration(self):
        return (self.end_time or time.perf_counter()) - self.start_time

class Tracer:
    """Builds a trace tree of spans."""
    def __init__(self):
        self.root = None
        self.stack = []
    def start_span(self, name):
        parent = self.stack[-1] if self.stack else None
        span = Span(name, parent_id=parent.span_id if parent else None,
                     trace_id=self.root.trace_id if self.root else None)
        if parent:
            parent.children.append(span)
        if self.root is None:
            self.root = span
        self.stack.append(span)
        return span
    def end_span(self):
        self.stack.pop().end()

# Build a small trace: root -> retrieve -> embed, vector_search -> generate.
tracer = Tracer()
root = tracer.start_span("rag_request")
root.set_attribute("user_query", "What is RAG?")
retrieve = tracer.start_span("retrieve")
embed = tracer.start_span("embed_query")
embed.set_attribute("tokens", 5)
time.sleep(0.01); tracer.end_span()  # embed
vec = tracer.start_span("vector_search")
vec.set_attribute("k", 3)
time.sleep(0.02); tracer.end_span()  # vector_search
tracer.end_span()  # retrieve
gen = tracer.start_span("generate")
gen.set_attribute("model", "gpt-4o-mini")
gen.set_attribute("tokens_in", 50)
gen.set_attribute("tokens_out", 100)
time.sleep(0.05); tracer.end_span()  # generate
tracer.end_span()  # rag_request

print(f"root span duration: {root.duration*1000:.1f} ms")
print(f"trace ID: {root.trace_id}")
print(f"root children: {[c.name for c in root.children]}")
```

**What this code does:**

- **Headline:** `import time, uuid`
- **Defines:** class `Span` — A single span in a trace: name, start/end, attributes, parent.; class `Tracer` — Builds a trace tree of spans.
- **Key imports:** `import time`, `from typing import Optional`
- **Notable calls:** `RAG`

> **Takeaway:** Tracing = a tree of spans with timing + attributes. Every
> tracing library implements this. The hard part is the backend (storage,
> query, UI).

> **What this does:** Renders the trace tree as text — parent spans, child
> spans, durations, attributes. The same shape you'd see in LangSmith or
> Jaeger.
> 
> **Why it matters:** the trace tree is the *debugging view* for LLM apps.
> When something goes wrong (slow response, bad answer, high cost), the trace
> tells you *which span* to investigate.
> 
> **What to watch for:** we render text. Real tools render interactive HTML
> with collapsible subtrees and attribute search.


### Code cell #2
```python
def render_span(span, depth=0):
    indent = "  " * depth
    dur = span.duration * 1000
    print(f"{indent}- {span.name} ({dur:.1f} ms)")
    for k, v in span.attributes.items():
        print(f"{indent}    {k} = {v}")
    for child in span.children:
        render_span(child, depth + 1)

print("=== TRACE ===")
render_span(tracer.root)
```

**What this code does:**

- **Headline:** `def render_span(span, depth=0):`
- **Defines:** def `render_span`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Demonstrates `langfuse` — the open-source LLMOps
> platform — for tracing. If `langfuse` isn't installed, we print the
> equivalent pattern.
> 
> **Why it matters:** langfuse (and LangSmith, Arize Phoenix) give you a
> production-grade trace UI without writing one. Just decorate your
> functions; the library builds the trace tree automatically.
> 
> **What to watch for:** langfuse requires a self-hosted server or cloud
> account. For local development, you can run it via docker.


### Code cell #3
```python
try:
    from langfuse import Langfuse, observe
    langfuse = Langfuse()
    print("langfuse installed; pattern would be:")
    print("  @observe()  # decorator that auto-traces the function")
    print("  def my_function(...): ...")
    print("  result = my_function(...)")
    print("  # trace visible in langfuse UI")
except ImportError:
    print("langfuse not installed — pip install langfuse")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from langfuse import Langfuse`

> **What this does:** Demonstrates OpenTelemetry for tracing — the
> industry-standard, language-agnostic tracing API. Most production LLM
> observability tools (LangSmith, Langfuse) can export to OTLP.
> 
> **Why it matters:** OpenTelemetry is the lingua franca of tracing. If you
> use it, your traces can flow to any backend (Jaeger, Tempo, Datadog,
> Honeycomb). You're not locked into one vendor.
> 
> **What to watch for:** OTel traces have *span events* and *span links* in
> addition to attributes. These let you record things like LLM retries and
> agent hand-offs without bloating the span count.


### Code cell #4
```python
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("demo")

    with tracer.start_as_current_span("rag_request") as span:
        span.set_attribute("user_query", "What is RAG?")
        with tracer.start_as_current_span("retrieve"):
            import time; time.sleep(0.02)
        with tracer.start_as_current_span("generate"):
            span.set_attribute("model", "gpt-4o-mini")
            time.sleep(0.05)
    print("OTel trace sent to console exporter")
except ImportError:
    print("opentelemetry not installed — pip install opentelemetry-api opentelemetry-sdk")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from opentelemetry import trace`, `from opentelemetry.sdk.trace import TracerProvider`, `from opentelemetry.sdk.trace.export import ConsoleSpanExporter`, `import time; time.sleep(0.02)`
- **Notable calls:** `RAG`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small set of RAG requests with varying
> complexity — single retrieval, multi-hop, multi-tool. We'll trace each
> and compare costs.
> 
> **Why it matters:** trace data is only useful if you compare across
> requests. This eval set gives you 5 traces to compare.
> 
> **What to watch for:** we manually tag each request with expected spans.
> Production tracing tools (LangSmith) auto-infer this from your code.


### Code cell #5
```python
trace_eval = [
    {"query": "What is RAG?",                "expected_spans": ["retrieve", "generate"]},
    {"query": "Compare RAG and fine-tuning", "expected_spans": ["retrieve", "retrieve", "generate"]},
    {"query": "Compute population of France times 3",
     "expected_spans": ["search", "calculator", "generate"]},
    {"query": "Summarise this article",      "expected_spans": ["generate"]},
    {"query": "What is 2+2?",                "expected_spans": ["generate"]},
]
print(f"trace eval size: {len(trace_eval)}")
for t in trace_eval:
    print(f"  Q: {t['query'][:50]:52s} expected: {t['expected_spans']}")
```

**What this code does:**

- **Headline:** `trace_eval = [`
- **Notable calls:** `RAG`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the trace tree we built as a horizontal
> bar chart: each span is a bar, length = duration, colour = depth.
> 
> **Why it matters:** the bar chart is the standard "trace view" in any
> production observability tool. Long spans are bottlenecks; wide spans
> are parallelism opportunities.
> 
> **What to watch for:** the chart should align to the *start time* (left
> edge), not the depth. A span that starts late but ends early is a child
> of an earlier span — alignment makes this obvious.


### Code cell #6
```python
import matplotlib.pyplot as plt

# Flatten the tree into a list of (span, depth, start, end).
spans = []
def flatten(span, depth=0, t0=root.start_time):
    spans.append((span, depth, span.start_time - t0, span.end_time - t0))
    for c in span.children:
        flatten(c, depth + 1, t0)
flatten(tracer.root)

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#3498db", "#27ae60", "#9b59b6", "#e74c3c", "#f39c12"]
for i, (span, depth, start, end) in enumerate(spans):
    ax.barh(len(spans) - i - 1, end - start, left=start, height=0.7,
            color=colors[depth % len(colors)], edgecolor="white")
    ax.text(start, len(spans) - i - 1, f" {span.name}", va="center", fontsize=9)
ax.set_yticks([])
ax.set_xlabel("time (s)")
ax.set_title("Trace tree as horizontal bar chart (coloured by depth)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Defines:** def `flatten`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Tracing doesn't evaluate models — it provides the data you use to evaluate
> *operations*. Common operational metrics:
> 
> - **P50/P95/P99 latency** — what's the user experience?
> - **Error rate** — what fraction of traces contain a failed span?
> - **Cost per request** — sum of token costs across all spans.
> - **Span count per trace** — proxy for pipeline complexity.

> **What this does:** Computes the four standard operational metrics from
> our trace tree.
> 
> **Why it matters:** these are the metrics you'd put on a production LLM
> dashboard. Without them, you don't know if your service is healthy.
> 
> **What to watch for:** P95 is more informative than P50 for latency —
> average users see P50, but the worst users see P95+.


### Code cell #7
```python
# Flatten the trace into a list of all spans.
all_spans = []
def collect(span):
    all_spans.append(span)
    for c in span.children:
        collect(c)
collect(tracer.root)

durations_ms = [s.duration * 1000 for s in all_spans]
token_count = sum(s.attributes.get("tokens_in", 0) + s.attributes.get("tokens_out", 0) for s in all_spans)

print(f"trace latency:    {tracer.root.duration*1000:.1f} ms total")
print(f"span count:        {len(all_spans)}")
print(f"span durations:    min={min(durations_ms):.1f} ms  max={max(durations_ms):.1f} ms")
print(f"total tokens:      {token_count}")
print(f"cost (mock $0.01/k): ${token_count * 0.01 / 1000:.4f}")
```

**What this code does:**

- **Headline:** `all_spans = []`
- **Defines:** def `collect`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a "retry" span when the LLM call fails — what does the trace tree look like?
> 2. Use langfuse or LangSmith to trace a real RAG request.
> 3. Compute P50/P95 latency over 100 mock requests.
> 4. Add a *cost* attribute to each span and compute the total trace cost.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a tracing decorator (`@trace`) that auto-creates spans for any function.
> 2. Build a trace inspector that flags any trace with > 5x normal latency.
> 3. Implement distributed tracing: a request that spans 2 services.
> 4. Add OpenTelemetry context propagation to an agent loop.
> 5. Build a dashboard that aggregates traces into p50/p95/cost trends.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between logging, metrics, and tracing?
> 2. What is a span, and what is a trace?
> 3. Why are traces tree-shaped rather than list-shaped?
> 4. What is OpenTelemetry, and why is it vendor-neutral?
> 5. How do you compute the cost of an LLM trace?
> 6. What are P50/P95/P99, and why is P95 more useful than mean?
> 7. How would you debug a slow LLM app using traces?
> 8. What is the difference between LangSmith, Langfuse, and Arize Phoenix?
