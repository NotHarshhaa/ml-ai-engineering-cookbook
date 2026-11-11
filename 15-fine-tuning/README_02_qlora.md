# Code Walkthrough — `02_qlora.ipynb`

> Module: **15-fine-tuning** | Notebook: **02_qlora.ipynb**
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

> # 02 — QLoRA
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
> ## 1. Concept Overview — QLoRA — 4-bit quantization + LoRA

> QLoRA combines two ideas: load the base model in 4-bit NF4 quantization
> (huge memory savings), then train LoRA adapters in fp16 (full gradient
> precision). The result: fine-tune a 7B LLM on a single 16 GB GPU.
> 
> **Beginner track:** understand the four-bit NormalFloat (NF4) format and
> why double quantization helps.
> **Advanced track:** also learn paged optimizers (QLoRA's trick for handling
> optimizer state OOM).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **NF4 (4-bit NormalFloat)**: a 4-bit data type whose values are optimally
> chosen for normally-distributed weights. Each weight is mapped to one of
> 16 values via $q = \text{argmin}_i |w - v_i|$.
> 
> **Double quantization**: the *quantization scales themselves* are
> quantized to 8-bit, saving an extra ~0.4 bits per parameter.
> 
> **Paged optimizers**: use NVIDIA unified memory to offload optimizer state
> to CPU when GPU memory is full. Prevents OOM during training spikes.
> 
> Total: a 7B model uses ~5 GB VRAM (vs 14 GB fp16, 28 GB fp32).

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements 4-bit quantization (round-to-nearest, not
> NF4) in pure NumPy. Shows the precision-vs-memory trade-off.
> 
> **Why it matters:** QLoRA's core is the same as our int8 quantizer, just
> with 16 levels instead of 256. The NF4 choice of *which* 16 levels is the
> innovation.
> 
> **What to watch for:** we use evenly-spaced levels. NF4 uses normally-
> distributed levels, which match the empirical weight distribution of real
> LLMs.


### Code cell #1
```python
import numpy as np

def quantize_int4(w, levels=16):
    """Symmetric 4-bit quantization with `levels` evenly-spaced values."""
    scale = np.abs(w).max() / (levels // 2 - 1)
    if scale == 0: scale = 1.0
    q = np.clip(np.round(w / scale), -(levels // 2), levels // 2 - 1).astype(np.int8)
    return q, scale

def nf4_levels():
    """The 16 NF4 levels — chosen for normally-distributed weights."""
    # Approximate the NF4 quantile levels of a standard normal.
    from scipy.stats import norm
    return norm.ppf(np.linspace(0, 1, 16)[1:-1])  # drop the ±inf endpoints

def quantize_nf4(w):
    """NF4: pick the nearest of the 16 NF4 levels (after standardising)."""
    levels = nf4_levels()
    scale = np.abs(w).max() / max(abs(levels[0]), abs(levels[-1]))
    if scale == 0: scale = 1.0
    norm_w = w / scale
    # Find nearest level for each weight.
    q = np.zeros_like(w, dtype=np.int8)
    for idx in np.ndindex(w.shape):
        q[idx] = np.argmin(np.abs(norm_w[idx] - levels))
    return q, scale, levels

def dequantize_nf4(q, scale, levels):
    return scale * levels[q]

# Demo: quantize a small weight matrix and measure error.
rng = np.random.default_rng(0)
W = rng.standard_normal((64, 128)) * 0.05   # realistic NN weight scale
q, scale, levels = quantize_nf4(W)
W_hat = dequantize_nf4(q, scale, levels)

print(f"original dtype: float32, size: {W.size * 4 / 1024:.1f} KB")
print(f"quantized dtype: int4 (as int8), size: {q.size / 1024:.1f} KB (8x smaller)")
print(f"max abs error: {np.abs(W - W_hat).max():.6f}")
print(f"relative error: {np.abs(W - W_hat).mean() / np.abs(W).mean():.4f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `quantize_int4` — Symmetric 4-bit quantization with `levels` evenly-spaced values.; def `nf4_levels` — The 16 NF4 levels — chosen for normally-distributed weights.; def `quantize_nf4` — NF4: pick the nearest of the 16 NF4 levels (after standardising).; def `dequantize_nf4`
- **Key imports:** `import numpy as np`, `from scipy.stats import norm`

> **Takeaway:** QLoRA's quantization is just int4 with NF4 levels. The
> innovation is *which* 16 values to use — NF4 picks them to match the
> weight distribution.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Demonstrates the standard QLoRA loading pattern in
> HuggingFace. If bitsandbytes isn't installed, we print the pattern.
> 
> **Why it matters:** this 4-line pattern is what every QLoRA fine-tuning
> script uses. Memorise it.
> 
> **What to watch for:** `bnb_4bit_compute_dtype=torch.float16` means the
> *forward* pass computes in fp16 (the 4-bit weights are dequantized on the
> fly). The backward pass is also in fp16 — only the *storage* is 4-bit.


### Code cell #2
```python
try:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("bitsandbytes installed; QLoRA pattern would be:")
    print("  bnb_config = BitsAndBytesConfig(")
    print("      load_in_4bit=True,")
    print("      bnb_4bit_quant_type='nf4',")
    print("      bnb_4bit_use_double_quant=True,")
    print("      bnb_4bit_compute_dtype=torch.bfloat16,")
    print("  )")
    print("  model = AutoModelForCausalLM.from_pretrained(")
    print("      'meta-llama/Llama-2-7b', quantization_config=bnb_config")
    print("  )")
    print("  # Then wrap with peft LoraConfig as in notebook 1")
except ImportError as e:
    print(f"bitsandbytes not installed ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import torch`, `from transformers import AutoModelForCausalLM`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small chat-format dataset for instruction
> tuning, formatted with a chat template. The format is what HuggingFace
> `trl.SFTTrainer` expects.
> 
> **Why it matters:** the chat template is what makes a base LLM behave like
> an assistant. Different models have different templates (Llama-2 vs
> Llama-3 vs Mistral) — get this wrong and the model produces nonsense.
> 
> **What to watch for:** the `system` message sets behaviour; the `user`
> message is the prompt; the `assistant` message is the expected output.
> Loss is computed only on the assistant turn.


### Code cell #3
```python
chat_data = [
    {"role": "system",    "content": "You are a helpful math tutor."},
    {"role": "user",      "content": "What is 12 * 7?"},
    {"role": "assistant", "content": "12 * 7 = 84."},
]
def format_chat(messages):
    """Format a chat as a single training string."""
    return "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)

text = format_chat(chat_data)
print("formatted chat (training text):")
print(text)
print(f"\nlength: {len(text)} chars")
```

**What this code does:**

- **Headline:** `chat_data = [`
- **Defines:** def `format_chat` — Format a chat as a single training string.

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the VRAM usage of full fine-tuning vs LoRA
> (fp16) vs QLoRA (4-bit) on a 7B model. The QLoRA bar is the one that fits
> on consumer hardware.
> 
> **Why it matters:** this chart is what convinced the world that LLM
> fine-tuning is democratisable. Before QLoRA, you needed a $10k GPU; after,
> you need a $500 one.
> 
> **What to watch for:** the QLoRA bar includes model weights (5 GB) + LoRA
> params (0.1 GB) + activations (4 GB) + optimiser states (0.5 GB). The
> exact split depends on batch size and sequence length.


### Code cell #4
```python
import matplotlib.pyplot as plt

methods = ["Full FT (fp32)", "LoRA (fp16)", "QLoRA (4-bit)"]
vram_gb = [80, 24, 8]   # approximate for a 7B model

plt.figure(figsize=(8, 4))
colors = ["#e74c3c", "#3498db", "#27ae60"]
bars = plt.bar(methods, vram_gb, color=colors, edgecolor="white")
plt.axhline(24, color="gray", linestyle="--", alpha=0.5, label="RTX 3090 (24 GB)")
plt.axhline(16, color="gray", linestyle=":", alpha=0.5, label="RTX 4080 (16 GB)")
plt.ylabel("VRAM required (GB)")
plt.title("VRAM needed to fine-tune a 7B model")
for bar, v in zip(bars, vram_gb):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f"{v} GB",
             ha="center", fontsize=10)
plt.legend()
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> QLoRA is evaluated on:
> 
> - **Quality vs full fine-tuning** — typically within 1% on standard benchmarks.
> - **VRAM usage** — should fit on the target GPU.
> - **Training time** — slower than LoRA (dequant overhead), but much faster than full FT on the same hardware.
> - **Inference latency** — after merging, identical to the base model.

> **What this does:** Simulates the typical quality-vs-VRAM trade-off of
> QLoRA vs LoRA vs full fine-tuning. Real numbers depend on the model and
> task.
> 
> **Why it matters:** QLoRA's small quality drop (<1% on most tasks) is the
> key finding that made it the standard. If your use case needs <0.5% drop,
> you may need fp16 LoRA; otherwise QLoRA is fine.
> 
> **What to watch for:** QLoRA's quality drop is bigger on small models
> (<3B). For 7B+, it's nearly free.


### Code cell #5
```python
# Mock results on a 7B model fine-tuned on a 10k instruction dataset.
results = [
    {"method": "Full FT",    "vram_gb": 80, "acc": 0.85, "time_hours": 12},
    {"method": "LoRA",       "vram_gb": 24, "acc": 0.84, "time_hours": 4},
    {"method": "QLoRA",      "vram_gb":  8, "acc": 0.84, "time_hours": 6},
]
print("method       VRAM     acc      time")
for r in results:
    print(f"{r['method']:10s} {r['vram_gb']:>5}GB {r['acc']:>6.2%} {r['time_hours']:>9.1f}h")
```

**What this code does:**

- **Headline:** `results = [`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Load a real 7B model with QLoRA and check VRAM usage with `torch.cuda.memory_allocated()`.
> 2. Try NF4 vs FP4 quantization — which is more accurate?
> 3. Disable `bnb_4bit_use_double_quant` — does the VRAM change?
> 4. Use paged optimizers (`optim=

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a custom NF4 quantizer with the exact levels from the paper.
> 2. Build a QLoRA training script using `trl.SFTTrainer` and run on a small dataset.
> 3. Implement adapter-only inference: load base + multiple LoRA adapters and switch.
> 4. Fine-tune a 7B model on a custom dataset with QLoRA in <8 GB VRAM.
> 5. Compare QLoRA to GPTQ quantization + LoRA.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What problem does QLoRA solve that LoRA alone doesn't?
> 2. What is NF4, and why is it better than uniform int4?
> 3. What is double quantization, and how much does it save?
> 4. What are paged optimizers, and when do they help?
> 5. How does QLoRA compare to GPTQ + LoRA?
> 6. Can you inference QLoRA models without merging? What are the trade-offs?
> 7. What VRAM does QLoRA need for a 70B model?
> 8. What are the quality trade-offs of QLoRA vs full fine-tuning?
