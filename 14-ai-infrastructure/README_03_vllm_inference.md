# Code Walkthrough — `03_vllm_inference.ipynb`

> Module: **14-ai-infrastructure** | Notebook: **03_vllm_inference.ipynb**
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

> # 03 — vLLM & Efficient Inference
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
> ## 1. Concept Overview — vLLM and efficient LLM inference

> vLLM is the production-grade LLM inference engine. Its key innovation is
> *PagedAttention* — a KV-cache management technique that reduces memory
> waste and enables 10-100x higher throughput than naive inference.
> 
> **Beginner track:** understand the three vLLM features: continuous
> batching, paged KV-cache, and tensor-parallel inference.
> **Advanced track:** also learn speculative decoding, prefix caching, and
> how to tune vLLM for your hardware.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Naive inference**: one request at a time, KV-cache allocated per
> request. Memory waste: up to 80% due to fragmentation.
> 
> **PagedAttention**: KV-cache is divided into fixed-size *pages* (like OS
> virtual memory). Pages are allocated on demand, shared across requests with
> the same prefix (e.g. system prompt), and de-allocated when the request
> ends. Memory waste drops to <5%.
> 
> **Continuous batching**: requests join/leave a batch dynamically. No need
> to wait for all requests in a batch to finish — total throughput goes up
> 10-100x.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Simulates PagedAttention in pure Python. We allocate
> KV-cache as pages, share pages between requests with the same prefix, and
> free pages when requests complete.
> 
> **Why it matters:** PagedAttention is the core of vLLM. Once you've seen
> the page-table data structure, every other LLM inference engine makes
> sense.
> 
> **What to watch for:** real vLLM pages are ~16 tokens; ours are ~4.
> Real vLLM allocates from a GPU-resident block pool; ours uses a Python
> list.


### Code cell #1
```python
class PagedKVCache:
    """Simulate vLLM-style paged KV cache."""
    def __init__(self, n_pages=64, page_size=4):
        self.page_size = page_size
        self.free_pages = list(range(n_pages))   # pool of free page IDs
        self.requests = {}  # req_id -> list of page IDs
    def allocate(self, req_id, n_tokens, shared_pages=None):
        """Allocate pages for a new request. Optionally share prefix pages."""
        if shared_pages:
            pages = list(shared_pages)   # share prefix
            n_tokens -= len(shared_pages) * self.page_size  # already allocated
        else:
            pages = []
        while n_tokens > 0 and self.free_pages:
            pages.append(self.free_pages.pop(0))
            n_tokens -= self.page_size
        self.requests[req_id] = pages
        return pages
    def free(self, req_id):
        for p in self.requests.get(req_id, []):
            self.free_pages.append(p)
        del self.requests[req_id]
    def used_pages(self):
        return sum(len(p) for p in self.requests.values())

cache = PagedKVCache(n_pages=64, page_size=4)
# Request 1: 12 tokens.
p1 = cache.allocate("req1", 12)
print(f"req1 pages: {p1}, total used: {cache.used_pages()}")
# Request 2 shares first 8 tokens with req1 (same prefix).
p2 = cache.allocate("req2", 16, shared_pages=p1[:2])  # share first 2 pages (8 tokens)
print(f"req2 pages: {p2}, total used: {cache.used_pages()}")
print(f"shared pages saved: {2} pages")
cache.free("req1")
print(f"after free req1, total used: {cache.used_pages()}")
```

**What this code does:**

- **Headline:** `class PagedKVCache:`
- **Defines:** class `PagedKVCache` — Simulate vLLM-style paged KV cache.

> **Takeaway:** PagedAttention = KV-cache as a page table. Sharing +
> lazy allocation gives huge memory savings and enables continuous batching.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Demonstrates the vLLM Python API. If vLLM isn't
> installed (or no GPU available), we print the pattern.
> 
> **Why it matters:** vLLM is the standard for LLM serving in 2024-2025.
> Knowing its API means you can deploy an LLM service in 5 lines.
> 
> **What to watch for:** `LLM(model=...)` loads the model; `llm.generate()` runs
> batched inference. The `SamplingParams` object controls temperature, top-p,
> and max tokens.


### Code cell #2
```python
try:
    from vllm import LLM, SamplingParams
    print("vLLM installed; pattern would be:")
    print("  llm = LLM(model='meta-llama/Llama-2-7b-hf', quantization='awq')")
    print("  sampling = SamplingParams(temperature=0.7, max_tokens=100)")
    print("  outputs = llm.generate(['Hello, my name is', 'The capital of France is'], sampling)")
    print("  for o in outputs: print(o.outputs[0].text)")
except ImportError:
    print("vllm not installed — pip install vllm  (requires CUDA GPU)")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from vllm import LLM`

> **What this does:** Compares naive batching (process all prompts, wait for
> all to finish) vs continuous batching (process each prompt, free resources
> as each completes). The second is much higher throughput.
> 
> **Why it matters:** continuous batching is the single biggest throughput
> innovation in modern LLM serving. Without it, GPU utilisation is ~20%;
> with it, ~80%.
> 
> **What to watch for:** in continuous batching, requests with different
> output lengths can share a batch. Naive batching must pad to the longest,
> wasting compute on the shorter ones.


### Code cell #3
```python
import time
import numpy as np

# Mock: 5 prompts with output lengths 10, 20, 30, 40, 50.
prompts = [10, 20, 30, 40, 50]

# Naive: process all at once, wait for the longest (50).
naive_time = max(prompts)  # max time = longest output

# Continuous: free each prompt as it finishes; new ones can join.
# Total time = sum of "useful" work divided by throughput (which is now higher).
continuous_time = sum(prompts) / 3  # rough: 3x throughput from batching

print(f"naive batch total time:      {naive_time} units")
print(f"continuous batch total time: {continuous_time:.1f} units")
print(f"throughput improvement: {naive_time / continuous_time:.1f}x")
```

**What this code does:**

- **Headline:** `import time`
- **Key imports:** `import time`, `import numpy as np`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a mock serving workload: 100 prompts of varying
> length, generated by 10 simulated users. This is the workload an LLM
> service sees in production.
> 
> **Why it matters:** real LLM services see highly variable request lengths.
> A workload with all 50-token prompts is easy; a workload with a 50:1 length
> ratio is hard. Test with the latter.
> 
> **What to watch for:** we mock the workload. Real workloads have bursty
> traffic patterns — peak hour traffic is 10-100x off-peak.


### Code cell #4
```python
import numpy as np
rng = np.random.default_rng(0)
n_prompts = 100

# Prompt lengths: log-normal (most short, a few very long).
prompt_lengths = rng.lognormal(mean=3.5, sigma=0.5, size=n_prompts).astype(int).clip(20, 2000)
# Output lengths: similar distribution.
output_lengths = rng.lognormal(mean=4.5, sigma=0.7, size=n_prompts).astype(int).clip(10, 1000)

print(f"prompt lengths: min={prompt_lengths.min()}, max={prompt_lengths.max()}, mean={prompt_lengths.mean():.0f}")
print(f"output lengths: min={output_lengths.min()}, max={output_lengths.max()}, mean={output_lengths.mean():.0f}")
print(f"max/min ratio: {output_lengths.max() / output_lengths.min():.0f}x  (high = hard workload)")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the cumulative request completion under naive
> vs continuous batching. Continuous completes short requests earlier; naive
> blocks all on the longest.
> 
> **Why it matters:** the "stair-step" pattern of naive batching vs the
> smooth curve of continuous batching is the visual signature of throughput
> improvement. Show this chart to anyone asking why vLLM is faster.
> 
> **What to watch for:** under continuous batching, the *first* response
> comes back in ~10 tokens worth of time. Under naive, the first response
> comes back in ~50 tokens worth of time. User-perceived latency is much
> lower with continuous.


### Code cell #5
```python
import matplotlib.pyplot as plt

# Sort output lengths ascending for the demo.
sorted_lens = sorted(output_lengths[:10])

# Naive: all complete at the max.
naive_completion = [max(sorted_lens)] * len(sorted_lens)

# Continuous: each completes at its own length.
continuous_completion = sorted_lens

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(range(1, 11), naive_completion, label="naive batching", where="mid", color="#e74c3c", lw=2)
ax.step(range(1, 11), continuous_completion, label="continuous batching", where="mid", color="#27ae60", lw=2)
ax.set_xlabel("request #"); ax.set_ylabel("completion time (tokens)")
ax.set_title("Naive vs continuous batching: cumulative completion")
ax.legend()
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Inference engines are evaluated by:
> 
> - **Throughput** — tokens/sec aggregated across all requests.
> - **Latency** — p50, p95, p99 of time-to-first-token and time-to-final-token.
> - **VRAM usage** — peak memory during serving.
> - **Quality** — quantization may degrade outputs; measure the drop.

> **What this does:** Computes throughput and average latency for our mock
> workload under naive and continuous batching.
> 
> **Why it matters:** these two numbers (throughput and latency) are the
> *operational* metrics for any LLM service. They tell you how many users
> you can serve and how good the experience is.
> 
> **What to watch for:** throughput and latency trade off. You can usually
> improve one at the cost of the other by changing the batch size.


### Code cell #6
```python
# Throughput = total tokens / total time.
total_tokens = sum(output_lengths)

# Naive: total time = sum of max output length per batch (assume 1 batch).
naive_total_time = max(output_lengths)
# Continuous: total time = sum of output lengths / parallelism (assume 8x).
continuous_total_time = sum(output_lengths) / 8

naive_throughput = total_tokens / naive_total_time
continuous_throughput = total_tokens / continuous_total_time

print(f"total output tokens: {total_tokens}")
print(f"naive:      time={naive_total_time}, throughput={naive_throughput:.1f} tok/s")
print(f"continuous: time={continuous_total_time:.0f}, throughput={continuous_throughput:.1f} tok/s")
print(f"improvement: {continuous_throughput / naive_throughput:.1f}x")
```

**What this code does:**

- **Headline:** `total_tokens = sum(output_lengths)`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Try vLLM with `enforce_eager=False` (CUDA graphs on) — measure the speedup.
> 2. Use tensor parallelism with `tensor_parallel_size=2` — does throughput double?
> 3. Compare vLLM to `transformers` (HuggingFace) generate on the same prompt.
> 4. Enable prefix caching (`enable_prefix_caching=True`) and benchmark a workload with shared prefixes.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Build a vLLM serving script that takes requests over HTTP with FastAPI.
> 2. Implement a request queue with priority levels.
> 3. Build a load test script that simulates 100 concurrent users.
> 4. Compare vLLM, TGI, and SGLang on the same workload.
> 5. Implement speculative decoding with a small draft model.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is PagedAttention, and why does it improve throughput?
> 2. What is continuous batching, and how does it differ from static batching?
> 3. What is the difference between vLLM, TGI, and SGLang?
> 4. What is prefix caching, and when does it help?
> 5. How does speculative decoding improve latency?
> 6. What is the KV-cache, and why does its size grow with sequence length?
> 7. How would you serve a 70B model on a single 80GB GPU?
> 8. What is tensor parallelism, and when is it preferable to pipeline parallelism?
