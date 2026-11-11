# Code Walkthrough — `02_quantization.ipynb`

> Module: **14-ai-infrastructure** | Notebook: **02_quantization.ipynb**
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

> # 02 — Model Quantization
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
> ## 1. Concept Overview — Model quantization

> Quantization reduces model precision from fp32 to fp16, int8, or int4.
> This shrinks the model 4-8x and dramatically speeds up inference, with
> minimal quality loss. Every LLM deployed at scale is quantized.
> 
> **Beginner track:** understand the difference between fp16 (no quality
> loss), int8 (small loss), and int4 (noticeable loss).
> **Advanced track:** also learn GPTQ, AWQ, and bitsandbytes (the standard
> LLM quantization libraries).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Quantization maps $w_\text{fp32}$ to $w_\text{int8}$ by:
> $$w_\text{int8} = \text{round}\!\left(\frac{w - \text{zero}}{\text{scale}}\right), \quad
> \text{scale} = \frac{\max(|w|)}{127}$$
> 
> Dequantization: $w \approx \text{scale} \cdot w_\text{int8} + \text{zero}$.
> 
> Per-tensor quant: one scale for the whole tensor. Per-channel quant:
> one scale per row. Per-channel is more accurate but slower.
> 
> Bits used: fp16 = 16, int8 = 8, int4 = 4. Memory savings: 2x, 4x, 8x vs fp32.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements per-tensor symmetric int8 quantization from
> scratch in NumPy. We quantize, dequantize, and measure the error.
> 
> **Why it matters:** quantization is a 10-line algorithm. The hard part is
> doing it *fast* on GPU and choosing the right scale per layer — but the
> core idea is what you see here.
> 
> **What to watch for:** `round` is non-differentiable, so quantization is
> applied *after* training (post-training quant) or simulated during
> training (quantization-aware training).


### Code cell #1
```python
import numpy as np

def quantize_int8(w):
    """Symmetric per-tensor int8 quantization."""
    scale = np.abs(w).max() / 127.0
    if scale == 0: scale = 1.0
    q = np.round(w / scale).astype(np.int8)
    return q, scale

def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale

# Quantize a random weight matrix.
rng = np.random.default_rng(0)
W = rng.standard_normal((64, 128)) * 0.1   # realistic NN weight scale
q, scale = quantize_int8(W)
W_hat = dequantize_int8(q, scale)

print(f"original dtype: {W.dtype}, size: {W.nbytes} bytes")
print(f"quantized dtype: {q.dtype}, size: {q.nbytes} bytes (4x smaller)")
print(f"max abs error: {np.abs(W - W_hat).max():.6f}")
print(f"relative error: {np.abs(W - W_hat).mean() / np.abs(W).mean():.4f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `quantize_int8` — Symmetric per-tensor int8 quantization.; def `dequantize_int8`
- **Key imports:** `import numpy as np`

> **Takeaway:** Quantization = `round(w / scale) * scale`. Memory savings
> are exactly the dtype ratio (4x for int8, 8x for int4). The error is the
> rounding noise.

> **What this does:** Compares per-tensor vs per-channel quantization on
> a weight matrix with very different row scales. Per-channel wins.
> 
> **Why it matters:** real LLM weights have wildly different per-channel
> scales (some rows are 100x larger than others). Per-tensor quantization
> wastes resolution on the small rows.
> 
> **What to watch for:** per-channel has one scale per row — slightly more
> memory for the scales, but much better accuracy. Almost always worth it.


### Code cell #2
```python
def quantize_per_channel(w):
    """Per-row (per-channel) int8 quantization."""
    scales = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.round(w / scales).astype(np.int8)
    return q, scales.ravel()

# Build a weight matrix with very different row scales.
W = rng.standard_normal((8, 32)) * np.linspace(0.01, 10.0, 8).reshape(-1, 1)
print(f"row scales (max abs): {np.abs(W).max(axis=1)}")

# Per-tensor quant: one scale for whole matrix.
q_pt, s_pt = quantize_int8(W)
err_pt = np.abs(W - dequantize_int8(q_pt, s_pt)).mean()

# Per-channel quant: one scale per row.
q_pc, s_pc = quantize_per_channel(W)
err_pc = np.abs(W - (q_pc.astype(np.float32) * s_pc.reshape(-1, 1))).mean()

print(f"per-tensor error: {err_pt:.6f}")
print(f"per-channel error: {err_pc:.6f}  (lower = better)")
```

**What this code does:**

- **Headline:** `def quantize_per_channel(w):`
- **Defines:** def `quantize_per_channel` — Per-row (per-channel) int8 quantization.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `bitsandbytes` for real LLM quantization — the
> standard library for loading LLMs in 4-bit (NF4) on a single GPU. If not
> installed, we print the pattern.
> 
> **Why it matters:** `bitsandbytes` is what makes `transformers`
> `BitsAndBytesConfig` work. Loading LLaMA-7B in 4-bit fits in 5 GB of VRAM
> instead of 14 GB (fp16) or 28 GB (fp32).
> 
> **What to watch for:** 4-bit quant has a noticeable quality drop on small
> models (< 3B params). For 7B+ models, the drop is usually <1% on standard
> benchmarks.


### Code cell #3
```python
try:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("bitsandbytes installed; pattern would be:")
    print("  bnb_config = BitsAndBytesConfig(")
    print("      load_in_4bit=True,")
    print("      bnb_4bit_quant_type='nf4',")
    print("      bnb_4bit_compute_dtype=torch.float16,")
    print("  )")
    print("  model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b', quantization_config=bnb_config)")
    print("  # uses ~5 GB VRAM instead of ~28 GB")
except ImportError as e:
    print(f"transformers or bitsandbytes not installed ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import torch`, `from transformers import AutoModelForCausalLM`

> **What this does:** Demonstrates PyTorch's built-in dynamic quantization —
> the simplest way to quantize a model for inference. If torch isn't
> available, we skip.
> 
> **Why it matters:** `torch.quantization.quantize_dynamic` quantizes the
> Linear and LSTM layers in one line, no retraining needed. It's the easiest
> way to get a 2x speedup on CPU inference.
> 
> **What to watch for:** dynamic quant only quantizes weights, not
> activations. For full int8 inference (weights + activations), use static
> quantization with calibration data.


### Code cell #4
```python
try:
    import torch
    import torch.quantization
    
    # A tiny model.
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10),
    ).eval()
    x = torch.randn(1, 64)
    
    # Original.
    out_fp32 = model(x)
    size_fp32 = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Quantized (int8 weights for Linear layers).
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out_int8 = qmodel(x)
    size_int8 = sum(p.numel() * (1 if p.qscheme() is not None else p.element_size())
                    for p in qmodel.parameters())  # rough
    
    print(f"fp32 output (first 5): {out_fp32[0, :5].tolist()}")
    print(f"int8 output (first 5): {out_int8[0, :5].tolist()}")
    print(f"max abs error: {(out_fp32 - out_int8).abs().max().item():.6f}")
except ImportError as e:
    print(f"torch not installed ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import torch`, `import torch.quantization`
- **Notable calls:** `nn.Linear`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Simulates the trade-off between quantization level
> and model accuracy. We generate mock numbers matching published results
> for LLaMA-7B on MMLU.
> 
> **Why it matters:** quantization is a trade-off curve, not a free lunch.
> Knowing the curve for *your* model on *your* benchmark is the only way to
> choose a quant level.
> 
> **What to watch for:** 4-bit quant on a 7B model is usually fine. On a 1B
> model, the accuracy drop is much more noticeable. Scale matters.


### Code cell #5
```python
quant_data = [
    {"precision": "fp32", "size_gb": 28, "mmlu_acc": 0.45, "vram_gb": 28},
    {"precision": "fp16", "size_gb": 14, "mmlu_acc": 0.45, "vram_gb": 14},
    {"precision": "int8",  "size_gb": 7,  "mmlu_acc": 0.44, "vram_gb": 8},
    {"precision": "int4 (NF4)", "size_gb": 3.5, "mmlu_acc": 0.43, "vram_gb": 5},
    {"precision": "int4 (GPTQ)", "size_gb": 3.5, "mmlu_acc": 0.42, "vram_gb": 5},
]
print("precision    size (GB)  MMLU acc  VRAM (GB)")
for q in quant_data:
    print(f"{q['precision']:12s} {q['size_gb']:>10.1f} {q['mmlu_acc']:>10.2%} {q['vram_gb']:>10.1f}")
```

**What this code does:**

- **Headline:** `quant_data = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the Pareto front of quantization: VRAM usage vs
> MMLU accuracy. The Pareto-optimal points are where you'd actually deploy.
> 
> **Why it matters:** the Pareto front shows the "honest" trade-off. A point
> off the front is strictly worse than one on it. Deployment picks a point on
> the front based on the latency/quality/VRAM budget.
> 
> **What to watch for:** fp16 and int4-NF4 are on the front. fp32 is
> dominated by fp16 (same accuracy, half the VRAM).


### Code cell #6
```python
import matplotlib.pyplot as plt

labels = [q["precision"] for q in quant_data]
vrams = [q["vram_gb"] for q in quant_data]
accs = [q["mmlu_acc"] for q in quant_data]

plt.figure(figsize=(8, 5))
plt.scatter(vrams, accs, s=120, color="steelblue", edgecolor="white")
for label, x, y in zip(labels, vrams, accs):
    plt.annotate(label, (x, y), xytext=(8, 5), textcoords="offset points", fontsize=9)
plt.xlabel("VRAM required (GB)")
plt.ylabel("MMLU accuracy")
plt.title("Quantization Pareto front for LLaMA-7B (mock data)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Quantization is evaluated by:
> 
> - **Quality drop** — accuracy / perplexity / task performance vs fp16.
> - **Memory reduction** — model size and VRAM usage.
> - **Latency improvement** — int8 should be 2-4x faster than fp16.
> - **Calibration cost** — how much data does quantization require?

> **What this does:** Measures the quality drop of our int8 quantization on
> a mock inference task: matmul against a random activation.
> 
> **Why it matters:** this is the *minimum* eval — measure the output
> difference between fp32 and int8. Real eval needs task-specific metrics
> (perplexity, accuracy) but max-abs-error is a good first sanity check.
> 
> **What to watch for:** an error >1% relative to the original magnitude
> suggests you need finer quantization (int8 per-channel, or int4 with a
> calibration dataset).


### Code cell #7
```python
# Compare fp32 matmul vs int8-quantized matmul.
W = rng.standard_normal((128, 256)) * 0.05
x = rng.standard_normal((1, 128))
y_fp32 = x @ W

q, scale = quantize_int8(W)
W_hat = dequantize_int8(q, scale)
y_int8 = x @ W_hat

err = np.abs(y_fp32 - y_int8).max()
mag = np.abs(y_fp32).max()
print(f"max abs output error: {err:.6f}")
print(f"max output magnitude: {mag:.4f}")
print(f"relative error: {err/mag:.4f}")
```

**What this code does:**

- **Headline:** `W = rng.standard_normal((128, 256)) * 0.05`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Quantize a real small model (DistilBERT) and measure the perplexity change.
> 2. Compare per-tensor vs per-channel quant at int8 — which is better?
> 3. Try GPTQ from `auto-gptq` on a small LLM.
> 4. Quantize to int4 with `bitsandbytes` and benchmark inference speed.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement quantization-aware training (QAT) — simulate quant during training.
> 2. Implement a per-channel int4 quantizer and compare to bitsandbytes NF4.
> 3. Build a quantization benchmark script that measures accuracy / VRAM / latency.
> 4. Implement dynamic vs static quantization and compare on a LSTM.
> 5. Use `autoawq` to quantize a LLaMA model and benchmark.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is quantization, and why does it work?
> 2. What is the difference between per-tensor and per-channel quantization?
> 3. What is GPTQ, and how does it differ from RTN (round-to-nearest)?
> 4. What is NF4, and why is it preferred for LLMs?
> 5. How does quantization-aware training work?
> 6. What is the accuracy / memory / latency trade-off for int4 vs int8?
> 7. How would you choose a quantization level for a 7B LLM in production?
> 8. What is calibration, and when do you need it?
