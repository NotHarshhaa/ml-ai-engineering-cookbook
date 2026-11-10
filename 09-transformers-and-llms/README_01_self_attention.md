# Code Walkthrough — `01_self_attention.ipynb`

> Module: **09-transformers-and-llms** | Notebook: **01_self_attention.ipynb**
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

> # 01 — Self-Attention from Scratch
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
> ## 1. Concept Overview — Self-attention from scratch

> Self-attention is the operation at the heart of every transformer. Each
> token in a sequence forms *queries*, *keys*, and *values* via three linear
> projections; the output for each token is a weighted average of the
> values, weighted by how well its query matches each key.
> 
> **Beginner track:** understand the Q, K, V intuition: each token asks a
> question (Q), and other tokens advertise themselves (K); matching tokens
> contribute their content (V).
> **Advanced track:** also learn why we divide by $\sqrt{d_k}$, and how
> multi-head attention differs from single-head (next notebook).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Scaled dot-product attention**:
> 
> $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$
> 
> where $Q, K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$.
> 
> The $\sqrt{d_k}$ scaling prevents the dot products from growing large in
> high dimensions, which would push softmax into the saturated region and
> produce vanishing gradients.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements scaled dot-product attention in pure NumPy.
> We generate random Q, K, V and produce the attention output + the weight
> matrix.
> 
> **Why it matters:** this is the *only* mathematical operation in a
> transformer that isn't a linear layer or layer-norm. Master it and you've
> mastered 70% of transformer internals.
> 
> **What to watch for:** the `softmax` is over the *last* axis of
> `Q @ K.T` — that's the *key* axis. Each query averages over keys.


### Code cell #1
```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (n, d_k), K: (n, d_k), V: (n, d_v) -> (n, d_v), (n, n)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # (n, n)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)     # (n, n) — each row sums to 1
    output = weights @ V                   # (n, d_v)
    return output, weights

# Tiny demo: 4 tokens, 8-dim queries/keys, 8-dim values.
rng = np.random.default_rng(0)
Q = rng.standard_normal((4, 8))
K = rng.standard_normal((4, 8))
V = rng.standard_normal((4, 8))
out, w = scaled_dot_product_attention(Q, K, V)
print("output shape:", out.shape)
print("weights row sums (should all be 1):", w.sum(axis=-1).round(4))
print("weights (first 2 rows):")
print(w[:2].round(3))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `softmax`; def `scaled_dot_product_attention` — Q: (n, d_k), K: (n, d_k), V: (n, d_v) -> (n, d_v), (n, n).
- **Key imports:** `import numpy as np`
- **Notable calls:** `softmax`

> **Takeaway:** Self-attention is one matrix multiply (QK^T), one softmax,
> one more matrix multiply (softmax @ V). That's it. The complexity is in
> the surrounding machinery — multi-head, positional encoding, layer norms.

> **What this does:** Demonstrates *causal* (autoregressive) attention by
> applying a triangular mask. Token $t$ can only attend to tokens $\le t$ —
> this is what GPT uses to prevent looking at future tokens.
> 
> **Why it matters:** the only difference between BERT (encoder, full
> attention) and GPT (decoder, causal attention) is this mask. Once you
> understand the mask, you understand the architectural split.
> 
> **What to watch for:** the mask is `True` where attention is *allowed*.
> Setting masked positions to `-inf` makes their softmax weight exactly 0.


### Code cell #2
```python
# Causal mask: token i can only attend to tokens j <= i.
n = 4
causal_mask = np.tril(np.ones((n, n), dtype=bool))   # lower-triangular True
print("causal mask:")
print(causal_mask.astype(int))

out_causal, w_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("\ncausal attention weights (note upper-triangle is 0):")
print(w_causal.round(3))
print("\nrow sums still 1:", w_causal.sum(axis=-1).round(4))
```

**What this code does:**

- **Headline:** `n = 4`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Implements scaled dot-product attention as a PyTorch
> `nn.Module`. This is *exactly* what `torch.nn.functional.scaled_dot_product_attention`
> does internally.
> 
> **Why it matters:** the PyTorch implementation is highly optimised (CUDA
> kernels, FlashAttention). But knowing what's inside it makes you a better
> transformer engineer — you'll spot bugs faster.
> 
> **What to watch for:** `F.softmax(..., dim=-1)` is over the *last* axis.
> If you accidentally pass `dim=-2`, attention weights become column-normalised
> and the model produces garbage.


### Code cell #3
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, n, d)
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)   # (B, H, n, n)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights

# Demo: batch=2, heads=1, seq=4, d=8.
Q = torch.randn(2, 1, 4, 8)
K = torch.randn(2, 1, 4, 8)
V = torch.randn(2, 1, 4, 8)
attn = ScaledDotProductAttention()
out, w = attn(Q, K, V)
print("output shape:", out.shape)
print("weights shape:", w.shape)
print("row sums (one head):", w[0, 0].sum(dim=-1).round(3))
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `ScaledDotProductAttention`
- **Key imports:** `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`
- **Notable calls:** `softmax`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a tiny "find-the-related-token" toy problem
> for self-attention: a sequence of random vectors where one pair has been
> manually correlated. A correctly trained attention layer should learn
> to put high weight between the correlated pair.
> 
> **Why it matters:** this is the smallest interesting test of attention.
> If your attention implementation is wrong, this task reveals it.
> 
> **What to watch for:** the correlation is added in the *value* of one
> token, but the attention is *learnt* from the keys and queries. Attention
> is what bridges the two.


### Code cell #4
```python
# Build a tiny synthetic sequence where token 2 should attend to token 5.
rng = np.random.default_rng(0)
n, d = 8, 16
seq = rng.standard_normal((n, d))
# Make token 2 strongly correlated with token 5.
seq[2] = seq[5] + 0.01 * rng.standard_normal(d)

# Use identity as Q, K, V (so attention is "vanilla" — just dot product).
out, w = scaled_dot_product_attention(seq, seq, seq)
print("attention weight from token 2 (row 2):")
print(w[2].round(3))
print(f"\nweight token 2 -> token 5: {w[2, 5]:.3f}  (should be highest)")
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(0)`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the attention weight matrix as a heatmap.
> Each row is a query; each column is a key; the colour is the attention
> weight.
> 
> **Why it matters:** this is the standard way to *see* what a transformer
> is doing. Real transformer visualisations use `bertviz` or `attentionviz`
> but the underlying matrix is the same.
> 
> **What to watch for:** in a randomly initialised transformer, the heatmap
> is roughly uniform. After training on a real task, you'll see structure:
> syntactic patterns in BERT, local patterns in early GPT layers, long-range
> in later ones.


### Code cell #5
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(w, cmap="viridis", aspect="auto")
axes[0].set_title("Full self-attention")
axes[0].set_xlabel("key position"); axes[0].set_ylabel("query position")
plt.colorbar(axes[0].imshow(w, cmap="viridis"), ax=axes[0], fraction=0.04)

# Causal version for comparison.
causal = np.tril(np.ones((n, n), dtype=bool))
_, w_c = scaled_dot_product_attention(seq, seq, seq, mask=causal)
axes[1].imshow(w_c, cmap="viridis", aspect="auto")
axes[1].set_title("Causal self-attention (GPT-style)")
axes[1].set_xlabel("key position"); axes[1].set_ylabel("query position")
plt.colorbar(axes[1].imshow(w_c, cmap="viridis"), ax=axes[1], fraction=0.04)
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Self-attention is evaluated by:
> 
> - **Downstream task accuracy** — the only metric that ultimately matters.
> - **Attention rollout / attention flow** — interpretability research.
> - **Effective receptive field** — how far attention reaches in practice.

> **What this does:** Computes attention *entropy* per query — how spread
> out is the attention distribution? A low entropy = focused attention; a
> high entropy = diffuse attention. This is a standard diagnostic.
> 
> **Why it matters:** if every token's attention is uniform (high entropy),
> the attention mechanism isn't learning anything. If it's too focused (low
> entropy on a single token), it may be ignoring useful context.
> 
> **What to watch for:** entropy is measured in *nats* (base-e). For a
> uniform distribution over $n$ tokens, the maximum entropy is $\log(n)$.


### Code cell #6
```python
def entropy(p, axis=-1, eps=1e-12):
    return -(p * np.log(p + eps)).sum(axis=axis)

entropies = entropy(w)
max_ent = np.log(n)
print("per-query entropy:", entropies.round(3))
print(f"max possible entropy (uniform): {max_ent:.3f}")
print(f"mean entropy: {entropies.mean():.3f}  (lower = more focused)")
```

**What this code does:**

- **Headline:** `def entropy(p, axis=-1, eps=1e-12):`
- **Defines:** def `entropy`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace `np.sqrt(d_k)` scaling with no scaling — does training become unstable?
> 2. Add a *padding mask* (zero out padding positions in the input).
> 3. Use a different temperature: `softmax(QK^T / T)` with T=0.5, 1.0, 2.0.
> 4. Compare attention entropy at initialisation vs after training on a tiny task.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement multi-head attention from scratch (next notebook does this).
> 2. Implement *linear* attention (no softmax — uses a kernel function).
> 3. Implement FlashAttention from scratch (just the algorithm, not the CUDA).
> 4. Build a 1-layer transformer block with self-attention + MLP.
> 5. Implement attention with relative positional encoding (e.g. ALiBi).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why is attention called *self*-attention? When is it not self?
> 2. Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention?
> 3. What is the difference between encoder attention and decoder (causal) attention?
> 4. What are Q, K, V, and why three matrices instead of one?
> 5. Why is attention invariant to permutation of the input sequence?
> 6. What is the time complexity of attention, and what does it imply for long sequences?
> 7. How does attention avoid the vanishing-gradient problem of RNNs?
> 8. What is FlashAttention, and what problem does it solve?
