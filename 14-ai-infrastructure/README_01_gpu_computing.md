# Code Walkthrough — `01_gpu_computing.ipynb`

> Module: **14-ai-infrastructure** | Notebook: **01_gpu_computing.ipynb**
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

> # 01 — GPU Computing Fundamentals
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
> ## 1. Concept Overview — GPU computing fundamentals

> GPUs are the workhorses of modern AI. A single H100 GPU has 80 GB of HBM3
> and ~2000 TOPS — making it ~10x faster than a high-end CPU for matrix
> multiplication. This notebook covers the GPU programming model: kernels,
> thread blocks, grids, and the memory hierarchy.
> 
> **Beginner track:** understand why GPUs are fast for ML (massive
> parallelism, high memory bandwidth).
> **Advanced track:** also learn CUDA kernels, tensor cores, and how to
> profile with `nsys` and `ncu`.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A GPU is a *SIMT* (single instruction, multiple thread) processor. A *kernel*
> is a function executed by thousands of threads in parallel. Threads are
> grouped into *blocks*; blocks into a *grid*.
> 
> For a matrix multiply $C = A B$ with $A, B \in \mathbb{R}^{n \times n}$:
> - CPU: $O(n^3)$ work, 1-16 threads → slow.
> - GPU: $O(n^3)$ work, $n^2$ threads → each thread computes one $C_{ij}$.
> Wall-clock drops from seconds to milliseconds.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Simulates the GPU thread-block model in pure Python —
> we "launch" a kernel as a grid of thread blocks, each block computing a
> tile of the output. This is the conceptual model behind every GPU ML
> library.
> 
> **Why it matters:** understanding the grid/block/thread hierarchy makes
> every PyTorch shape error and every CUDA OOM message comprehensible. The
> GPU is not magic; it's just very parallel.
> 
> **What to watch for:** we use Python threads to simulate GPU threads.
> Real GPUs run thousands of threads in parallel on SIMD lanes; Python's GIL
> prevents true parallelism here — this is just conceptual.


### Code cell #1
```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Simulate a GPU kernel: each "thread" computes one element of C = A @ B.
def matmul_kernel(args):
    i, j, A, B = args
    n = A.shape[1]
    s = 0.0
    for k in range(n):
        s += A[i, k] * B[k, j]
    return (i, j, s)

def gpu_style_matmul(A, B, threads_per_block=16):
    """Simulate a GPU matmul: launch a grid of thread blocks."""
    n_rows, n_cols = A.shape[0], B.shape[1]
    C = np.zeros((n_rows, n_cols))
    # Build the work list: one entry per output element.
    work = [(i, j, A, B) for i in range(n_rows) for j in range(n_cols)]
    # Simulate "blocks" of threads_per_block.
    with ThreadPoolExecutor(max_workers=threads_per_block) as ex:
        results = list(ex.map(matmul_kernel, work))
    for i, j, val in results:
        C[i, j] = val
    return C

rng = np.random.default_rng(0)
A = rng.standard_normal((32, 32))
B = rng.standard_normal((32, 32))
C = gpu_style_matmul(A, B, threads_per_block=16)
C_ref = A @ B
print("max abs error:", np.abs(C - C_ref).max())
print("output shape:", C.shape)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `matmul_kernel`; def `gpu_style_matmul` — Simulate a GPU matmul: launch a grid of thread blocks.
- **Key imports:** `import numpy as np`, `from concurrent.futures import ThreadPoolExecutor`

> **Takeaway:** GPU programming = mapping each output element to a thread,
> grouping threads into blocks, and launching a grid. PyTorch/CUDA handle
> this automatically for matmul — you only write kernels for custom ops.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Demonstrates PyTorch's GPU API. If a CUDA GPU isn't
> available, the code runs on CPU (slower but identical).
> 
> **Why it matters:** the `tensor.to(device)` API is the entire PyTorch GPU
> interface. Everything else is automatic.
> 
> **What to watch for:** `device="cuda"` requires NVIDIA drivers + CUDA.
> On Mac, use `device="mps"`; on CPU, fall back to `device="cpu"`.


### Code cell #2
```python
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# Move a tensor to GPU.
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

# Time the GPU matmul.
torch.cuda.synchronize() if device == "cuda" else None
t0 = time.perf_counter()
for _ in range(10):
    _ = x @ y
torch.cuda.synchronize() if device == "cuda" else None
t1 = time.perf_counter()
print(f"avg matmul time: {(t1-t0)/10*1000:.2f} ms")
print(f"output device: {(x @ y).device}")
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`, `import time`

> **What this does:** Demonstrates mixed-precision training with
> `torch.autocast` — using float16 for matmuls and float32 for reductions.
> This is 2-3x faster on modern GPUs and halves memory.
> 
> **Why it matters:** mixed precision is the standard for large-model
> training. Without it, training a 7B model on a single GPU is impossible.
> 
> **What to watch for:** autocast only converts inside its `with` block.
> Outside, operations run at the default dtype (float32). Don't manually
> cast inside autocast — let it decide.


### Code cell #3
```python
model = torch.nn.Linear(1000, 1000).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Mixed-precision training step.
opt.zero_grad()
with torch.autocast(device_type=device.split("_")[0] if "_" in device else device, dtype=torch.float16):
    out = model(x)
    loss = loss_fn(out, y)
loss.backward()
opt.step()
print(f"loss: {loss.item():.4f}  (mixed precision on {device})")
```

**What this code does:**

- **Headline:** `model = torch.nn.Linear(1000, 1000).to(device)`
- **Notable calls:** `nn.Linear`, `MSELoss`, `Adam`, `backward`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Benchmarks CPU vs GPU (if available) at multiple matrix
> sizes. This is the standard "is my GPU worth it?" test.
> 
> **Why it matters:** the speedup depends on matrix size. Small matrices are
> memory-bound and may be faster on CPU; large matrices are compute-bound and
> GPU wins by 10-100x.
> 
> **What to watch for:** at n=100, GPU may be slower than CPU (launch
> overhead dominates). At n=5000, GPU wins by ~50x.


### Code cell #4
```python
import time
import matplotlib.pyplot as plt

sizes = [100, 500, 1000, 2000, 4000]
cpu_times = []
gpu_times = []

for n in sizes:
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    # CPU.
    t0 = time.perf_counter()
    for _ in range(3): _ = a @ b
    cpu_times.append((time.perf_counter() - t0) / 3 * 1000)
    # GPU (if available).
    if device == "cuda":
        a_gpu = a.to(device); b_gpu = b.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3): _ = a_gpu @ b_gpu
        torch.cuda.synchronize()
        gpu_times.append((time.perf_counter() - t0) / 3 * 1000)
    else:
        gpu_times.append(None)

for i, n in enumerate(sizes):
    print(f"  n={n:5d}  CPU={cpu_times[i]:8.2f} ms  GPU={gpu_times[i] or 'n/a':>8}")
```

**What this code does:**

- **Headline:** `import time`
- **Key imports:** `import time`, `import matplotlib.pyplot as plt`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots CPU vs GPU matmul time across matrix sizes.
> The crossover point is where the GPU becomes worth using.
> 
> **Why it matters:** this is the chart you'd show a team deciding whether
> to invest in GPU infrastructure. The answer depends on the workload's
> matrix sizes.
> 
> **What to watch for:** on CPU, time grows as $n^3$. On GPU, it also grows
> as $n^3$ but with a much smaller constant. The gap *widens* with size.


### Code cell #5
```python
plt.figure(figsize=(8, 4))
plt.plot(sizes, cpu_times, "o-", label="CPU", color="steelblue")
if all(g is not None for g in gpu_times):
    plt.plot(sizes, gpu_times, "s-", label="GPU", color="orange")
plt.xlabel("matrix size n"); plt.ylabel("matmul time (ms)")
plt.yscale("log")
plt.legend()
plt.title("CPU vs GPU matmul time (log scale)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `plt.figure(figsize=(8, 4))`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> GPU performance is evaluated by:
> 
> - **FLOPS achieved** — peak vs sustained (usually 30-60% of peak).
> - **Memory bandwidth** — peak vs sustained.
> - **Time-to-solution** — the only metric that matters to users.
> - **Throughput** — requests/sec for serving.

> **What this does:** Computes achieved FLOPS for our matmul at n=2000.
> 
> **Why it matters:** GPUs are advertised by peak FLOPS, but you'll only
> see 30-60% in practice. Measuring achieved FLOPS tells you how well your
> code uses the hardware.
> 
> **What to watch for:** `2 * n^3` is the FLOP count for square matmul
> (n^3 multiplications + n^3 additions).


### Code cell #6
```python
n = 2000
flops = 2 * n ** 3
cpu_time = cpu_times[-1] / 1000  # seconds
print(f"matmul FLOPs: {flops:.2e}")
print(f"CPU time: {cpu_time:.3f} s")
print(f"CPU TFLOPS: {flops / cpu_time / 1e12:.2f}")
if all(g is not None for g in gpu_times):
    gpu_time = gpu_times[-1] / 1000
    print(f"GPU time: {gpu_time:.3f} s")
    print(f"GPU TFLOPS: {flops / gpu_time / 1e12:.2f}")
```

**What this code does:**

- **Headline:** `n = 2000`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace `torch.randn` with `torch.randn(dtype=torch.float16)` — does it change speed?
> 2. Use `torch.bmm` (batched matmul) on a 3-D tensor. How does it scale?
> 3. Profile with `torch.profiler` — where is the time going?
> 4. Try `torch.compile(model)` — does it speed up the training step?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Write a custom CUDA kernel with `numba.cuda` for element-wise square.
> 2. Implement a fused attention kernel and compare to PyTorch.
> 3. Profile a training step with `nsys` and identify the bottleneck.
> 4. Implement gradient accumulation to simulate a larger effective batch.
> 5. Build a multi-GPU training step with `DistributedDataParallel`.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why are GPUs faster than CPUs for ML?
> 2. What is the SIMT execution model?
> 3. What is a thread block, and why does it matter?
> 4. What is mixed precision, and what are its trade-offs?
> 5. How would you choose between fp16, bf16, and fp32?
> 6. What is the difference between `torch.cuda.synchronize()` and `torch.no_grad()`?
> 7. How do you measure achieved FLOPS?
> 8. What is `torch.compile`, and what does it actually compile?
