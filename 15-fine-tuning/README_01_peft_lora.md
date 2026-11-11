# Code Walkthrough — `01_peft_lora.ipynb`

> Module: **15-fine-tuning** | Notebook: **01_peft_lora.ipynb**
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

> # 01 — PEFT & LoRA
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
> ## 1. Concept Overview — PEFT and LoRA

> Full fine-tuning of a 7B LLM requires ~40 GB of GPU memory for optimiser
> states — out of reach for most developers. *Parameter-Efficient
> Fine-Tuning* (PEFT) freezes most of the model and trains only a small set
> of *adapter* weights. LoRA (Low-Rank Adaptation) is the most popular PEFT
> method: ~0.5% of the parameters, same accuracy, trainable on a single GPU.
> 
> **Beginner track:** learn the LoRA decomposition $W = W_0 + BA$ where
> $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll d$.
> **Advanced track:** also learn QLoRA, AdaLoRA, and how to choose the rank.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **LoRA**: replace a weight update $\Delta W \in \mathbb{R}^{d \times k}$ with
> a low-rank factorisation $\Delta W = B A$, where $B \in \mathbb{R}^{d \times r}$,
> $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.
> 
> Trainable params: $r(d + k)$ instead of $dk$. For $d = k = 4096$, $r = 8$:
> 65K params vs 16M — a 250x reduction.
> 
> Merge for inference: $W_\text{merged} = W_0 + \alpha B A / r$, where $\alpha$
> is a scaling hyperparameter (typically $\alpha = 2r$).
> 
> **QLoRA**: same idea but the base model $W_0$ is loaded in 4-bit NF4. The
> LoRA adapters are still trained in fp16/bf16. Memory: 4 GB instead of 14 GB
> for a 7B model.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements LoRA on a single `nn.Linear` layer from
> scratch in PyTorch. We show how to apply it to any linear layer and how to
> merge the adapters back into the base weight for inference.
> 
> **Why it matters:** LoRA looks like magic until you've written it. Then
> you see it's just two small matrices, a forward pass that adds BA to the
> linear output, and a backward pass that only updates B and A.
> 
> **What to watch for:** `B` is initialised to zero, `A` with random small
> values. This guarantees the adapter is a no-op at the start of training —
> the model behaves exactly like the base until you train.


### Code cell #1
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """A linear layer with a LoRA adapter: y = W x + B A x."""
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        # Freeze the base weight.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in = base.in_features
        d_out = base.out_features
        # LoRA matrices: A (d_in x rank), B (rank x d_out).
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)   # small random init
        self.B = nn.Parameter(torch.zeros(rank, d_out))          # zero init: BA = 0
        self.scaling = alpha / rank

    def forward(self, x):
        # Base (frozen) + LoRA (trainable).
        return self.base(x) + (x @ self.A @ self.B) * self.scaling

    def merge(self):
        """Merge the LoRA adapter into the base weight for inference."""
        with torch.no_grad():
            self.base.weight.add_(self.scaling * (self.A @ self.B))

# Demo: create a linear layer + LoRA adapter.
torch.manual_seed(0)
base = nn.Linear(64, 32)
lora_layer = LoRALinear(base, rank=4, alpha=8)

x = torch.randn(1, 64)
y_before = lora_layer(x)
print(f"base params: {sum(p.numel() for p in base.parameters()):,} (frozen)")
print(f"LoRA params: {lora_layer.A.numel() + lora_layer.B.numel():,} (trainable)")
print(f"forward output: {y_before[0, :3].tolist()}")

# After merging, the LoRA layer behaves identically but uses only base weight.
lora_layer.merge()
y_after = lora_layer(x)  # uses merged base weight
print(f"after merge: {y_after[0, :3].tolist()}")
print(f"diff: {(y_before - y_after).abs().max().item():.6f}  (should be ~0)")
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `LoRALinear` — A linear layer with a LoRA adapter: y = W x + B A x.
- **Key imports:** `import torch`, `import torch.nn as nn`
- **Notable calls:** `nn.Linear`

> **Takeaway:** LoRA = `(x @ A @ B) * scaling` added to a frozen linear
> layer. Two small matrices instead of one big one. Merge into the base
> weight for inference to recover the original speed.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses the `peft` library — the standard PEFT
> implementation from HuggingFace. We show the pattern for wrapping a
> transformer model with LoRA adapters.
> 
> **Why it matters:** `peft` is the de-facto standard for LoRA in 2024-2025.
> It integrates with `transformers`, `trl`, and most fine-tuning scripts.
> 
> **What to watch for:** `target_modules=[just the attention Q and V matrices — the standard choice for LLMs. Other
> choices (e.g. all linear) train more params but may improve quality.


### Code cell #2
```python
try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print("peft installed; pattern would be:")
    print("  model = AutoModelForCausalLM.from_pretrained(...)")
    print("  lora_config = LoraConfig(")
    print("      r=8, lora_alpha=16, lora_dropout=0.05,")
    print("      target_modules=['q_proj', 'v_proj'],")
    print("      task_type='CAUSAL_LM',")
    print("  )")
    print("  model = get_peft_model(model, lora_config)")
    print("  model.print_trainable_parameters()")
    print("  # ~0.5% of total params are trainable")
except ImportError as e:
    print(f"peft not installed ({e}) — pip install peft")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from peft import LoraConfig`, `from transformers import AutoModelForCausalLM`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a tiny instruction-tuning dataset: 10 (prompt,
> completion) pairs. This is the smallest setup on which you can demonstrate
> SFT (supervised fine-tuning).
> 
> **Why it matters:** SFT is what turns a base LLM into an assistant. The
> dataset shape is universal: (prompt, completion) pairs in a chat template.
> 
> **What to watch for:** real SFT datasets have 1k-100k examples. Our 10
> is for demonstrating the code; not enough to actually teach the model.


### Code cell #3
```python
sft_data = [
    {"prompt": "What is 2 + 2?",          "completion": "2 + 2 = 4"},
    {"prompt": "Capital of France?",       "completion": "Paris"},
    {"prompt": "Translate 'hello' to French.", "completion": "bonjour"},
    {"prompt": "Reverse 'abc'.",           "completion": "cba"},
    {"prompt": "What color is the sky?",   "completion": "Blue"},
    {"prompt": "Square root of 16?",        "completion": "4"},
    {"prompt": "Author of 'Hamlet'?",     "completion": "William Shakespeare"},
    {"prompt": "What is Python?",          "completion": "A programming language"},
    {"prompt": "Largest planet?",           "completion": "Jupiter"},
    {"prompt": "Boiling point of water?",   "completion": "100 degrees Celsius"},
]
print(f"SFT dataset size: {len(sft_data)}")
for ex in sft_data[:3]:
    print(f"  Q: {ex['prompt']}  A: {ex['completion']}")
```

**What this code does:**

- **Headline:** `sft_data = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises how the LoRA rank affects trainable
> parameter count and (mock) accuracy. The Pareto front shows where the rank
> stops helping.
> 
> **Why it matters:** choosing `r` is the main LoRA hyperparameter. Too low
> (r=1) underfits; too high (r=64) approaches full fine-tuning cost with no
> extra accuracy.
> 
> **What to watch for:** the knee of the curve is usually at r=8 or r=16.
> Going higher only helps for very hard tasks or very large models.


### Code cell #4
```python
import matplotlib.pyplot as plt

ranks = [1, 2, 4, 8, 16, 32, 64, 128]
# Mock: trainable params = 2 * rank * d (with d = 4096), accuracy grows with sqrt(rank).
import numpy as np
d = 4096
params = [2 * r * d for r in ranks]
# Mock accuracy: log(rank) scaled, capped at 0.95.
accs = [min(0.95, 0.85 + 0.05 * np.log2(r) / np.log2(128)) for r in ranks]

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(ranks, params, "o-", color="steelblue", label="trainable params")
ax1.set_xlabel("LoRA rank r"); ax1.set_ylabel("trainable params", color="steelblue")
ax1.set_yscale("log")
ax2 = ax1.twinx()
ax2.plot(ranks, accs, "s-", color="orange", label="downstream accuracy")
ax2.set_ylabel("downstream accuracy (mock)", color="orange")
plt.title("LoRA rank vs trainable params vs accuracy")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import numpy as np`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> LoRA / PEFT is evaluated by:
> 
> - **Downstream task accuracy** — same as the base model.
> - **Trainable parameter count** — should be <1% of total.
> - **Training memory** — should fit on a single GPU.
> - **Inference latency** — after merging, should match the base model.
> - **Adapter size** — typical 10-100 MB per adapter; easy to swap.

> **What this does:** Compares the trainable parameter count of full
> fine-tuning vs LoRA at several ranks. The reduction is dramatic.
> 
> **Why it matters:** this is the headline number for LoRA — 0.5% of params
> for ~99% of the quality. It's why PEFT has become the default for LLM
> fine-tuning.
> 
> **What to watch for:** `trainable %` is calculated against total params
> (base + adapter). The base params are frozen but still take VRAM during
> training (for activations).


### Code cell #5
```python
def lora_param_count(d_in, d_out, r):
    """Trainable params for a LoRA-adapted linear layer."""
    return r * (d_in + d_out)

# A 7B model: ~4096x4096 weights per layer, 32 layers.
d = 4096
n_layers = 32
full_params = d * d * n_layers

for r in [1, 4, 8, 16, 64]:
    lora_params = lora_param_count(d, d, r) * n_layers * 2  # Q and V per layer
    pct = 100 * lora_params / full_params
    print(f"  r={r:3d}: LoRA params = {lora_params:>12,}  ({pct:.2f}% of full)")
```

**What this code does:**

- **Headline:** `def lora_param_count(d_in, d_out, r):`
- **Defines:** def `lora_param_count` — Trainable params for a LoRA-adapted linear layer.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace `target_modules=[2. Try rank 4 vs 16 vs 64 on a small fine-tuning task — which is best?
> 3. Add `lora_dropout=0.1` — does it regularise?
> 4. Use QLoRA: load the base model in 4-bit with `bitsandbytes` and train.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement LoRA from scratch on a `nn.Conv2d` layer for vision fine-tuning.
> 2. Build a multi-adapter inference server: load multiple LoRAs and switch at runtime.
> 3. Implement AdaLoRA — adaptively allocate rank across layers.
> 4. Implement adapter merging: combine two LoRAs trained on different tasks.
> 5. Quantize a LoRA-merged model to int4 and benchmark.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is PEFT, and why is it useful?
> 2. How does LoRA work, in one paragraph?
> 3. Why is B initialised to zero in LoRA?
> 4. What is QLoRA, and how does it differ from LoRA?
> 5. How do you choose the LoRA rank?
> 6. What does `target_modules` control, and why does it matter?
> 7. How do you merge LoRA adapters back into the base model?
> 8. What is the difference between LoRA and full fine-tuning in quality?
