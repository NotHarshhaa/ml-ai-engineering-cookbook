# Code Walkthrough — `04_attention_mechanism.ipynb`

> Module: **08-natural-language-processing** | Notebook: **04_attention_mechanism.ipynb**
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

> # 04 — Attention Mechanism
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
> ## 1. Concept Overview — Attention mechanism

> Attention lets a sequence model *look back* at earlier tokens when producing
> each output. Originally proposed for machine translation (Bahdanau 2014) to
> fix LSTMs' bottleneck, attention became the foundation of transformers.
> 
> **Beginner track:** understand the additive (Bahdanau) and multiplicative
> (Luong) attention formulas. Both compute a weighted average of context
> vectors.
> **Advanced track:** also learn multi-head attention (covered in Module 09)
> and why dot-product attention beats additive at scale.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Bahdanau (additive) attention**:
> $$e_{t,i} = v^\top \tanh(W_s s_{t-1} + W_h h_i)$$
> $$\alpha_{t,i} = \text{softmax}(e_{t,i})_i$$
> $$c_t = \sum_i \alpha_{t,i} h_i$$
> 
> **Luong (multiplicative) attention**:
> $$e_{t,i} = s_t^\top W h_i, \quad \alpha = \text{softmax}(e), \quad c = \sum \alpha_i h_i$$
> 
> Both compute a weighted average of encoder hidden states, where weights
> are learned to focus on the *right* input tokens for each output.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements Bahdanau-style additive attention in pure
> NumPy. We compute attention weights and a context vector for a tiny
> encoder-decoder pair.
> 
> **Why it matters:** once you can implement attention by hand, you'll see
> that it's just three linear layers + a softmax + a weighted sum. The same
> operation underlies every transformer.
> 
> **What to watch for:** `softmax` over the *input sequence* dimension (axis=1
> in our (batch, seq, hidden) convention). Not over the batch.


### Code cell #1
```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def bahdanau_attention(h_enc, s_dec, W_s, W_h, v):
    """h_enc: (T, H), s_dec: (H,), W_s: (H, A), W_h: (H, A), v: (A,)."""
    # Project encoder hidden states and decoder state to a shared attention space.
    proj_h = h_enc @ W_h                 # (T, A)
    proj_s = s_dec @ W_s                 # (A,)
    # Score: v^T tanh(proj_h + proj_s).
    scores = np.tanh(proj_h + proj_s) @ v   # (T, A) @ (A,) -> (T,)
    weights = softmax(scores)               # (T,)
    context = weights @ h_enc                # (H,)
    return context, weights

# Tiny demo: 5 encoder hidden states, 8-dim each.
rng = np.random.default_rng(0)
T, H, A = 5, 8, 4
h_enc = rng.standard_normal((T, H))
s_dec = rng.standard_normal(H)
W_s = rng.standard_normal((H, A)) * 0.1
W_h = rng.standard_normal((H, A)) * 0.1
v   = rng.standard_normal(A) * 0.1

context, weights = bahdanau_attention(h_enc, s_dec, W_s, W_h, v)
print("attention weights:", weights.round(3))
print("context vector (first 4 dims):", context[:4].round(3))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `softmax`; def `bahdanau_attention` — h_enc: (T, H), s_dec: (H,), W_s: (H, A), W_h: (H, A), v: (A,).
- **Key imports:** `import numpy as np`
- **Notable calls:** `softmax`

> **Takeaway:** Attention = "weighted average of encoder states, weights
> learned to focus on relevant inputs." That's it. Everything else is engineering.

> **What this does:** Implements Luong (dot-product) attention. Simpler than
> Bahdanau — just one weight matrix and a dot product.
> 
> **Why it matters:** dot-product attention is what *scaled dot-product*
> attention (the transformer kind) builds on. Learn this and you're 90% of
> the way to understanding transformers.
> 
> **What to watch for:** the `W` matrix projects the encoder and decoder to a
> shared space before the dot product. Without it, attention is symmetric in
> ways that hurt learning.


### Code cell #2
```python
def luong_attention(h_enc, s_dec, W):
    """h_enc: (T, H), s_dec: (H,), W: (H, H)."""
    # Score: s_dec^T W h_i for each i.
    scores = (s_dec @ W) @ h_enc.T   # (T,)
    weights = softmax(scores)
    context = weights @ h_enc
    return context, weights

W = rng.standard_normal((H, H)) * 0.1
ctx, w = luong_attention(h_enc, s_dec, W)
print("Luong attention weights:", w.round(3))
print("context vector (first 4 dims):", ctx[:4].round(3))
```

**What this code does:**

- **Headline:** `def luong_attention(h_enc, s_dec, W):`
- **Defines:** def `luong_attention` — h_enc: (T, H), s_dec: (H,), W: (H, H).
- **Notable calls:** `softmax`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds an attention layer as a PyTorch `nn.Module`.
> This is the kind of code you'd write when implementing a research paper that
> uses Bahdanau attention on top of an LSTM.
> 
> **Why it matters:** understanding this code is the bridge from
> encoder-decoder RNNs to transformers. Multi-head attention (next module) is
> just this code, vectorised.
> 
> **What to watch for:** `torch.bmm` does batched matrix multiply — same as
> `np.einsum('bth,bh->bt')`. Use it instead of for-loops over the batch.


### Code cell #3
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, h_enc, s_dec):
        # h_enc: (B, T, H), s_dec: (B, H)
        proj_h = self.W_h(h_enc)                 # (B, T, A)
        proj_s = self.W_s(s_dec).unsqueeze(1)    # (B, 1, A)
        scores = self.v(torch.tanh(proj_h + proj_s)).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)     # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1)  # (B, H)
        return context, weights

attn = BahdanauAttention(hidden_dim=8, attn_dim=4)
h_enc = torch.randn(3, 5, 8)   # batch=3, seq=5, hidden=8
s_dec = torch.randn(3, 8)
context, weights = attn(h_enc, s_dec)
print("context shape:", context.shape)   # (3, 8)
print("weights shape:", weights.shape)   # (3, 5)
print("weights sum per batch:", weights.sum(dim=-1).round(3))
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `BahdanauAttention`
- **Key imports:** `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`
- **Notable calls:** `nn.Linear`, `softmax`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Generates a tiny sequence-to-sequence task: reverse a
> 5-digit sequence. We'll use it to test attention.
> 
> **Why it matters:** sequence reversal is the *hello world* of seq2seq.
> An LSTM without attention struggles; with attention it's trivial. The
> difference is the whole point of attention.
> 
> **What to watch for:** for this demo we don't even train — we just look
> at the data structure to understand the task.


### Code cell #4
```python
import numpy as np

rng = np.random.default_rng(0)

def make_seq(n_samples=200, seq_len=5, vocab=10):
    X = rng.integers(1, vocab, (n_samples, seq_len))
    # Reverse each row.
    Y = X[:, ::-1].copy()
    return X, Y

X, Y = make_seq(200, 5, 10)
print("first 3 (X, Y) pairs:")
for i in range(3):
    print(f"  X={X[i].tolist()}  Y={Y[i].tolist()}")
print("shapes:", X.shape, Y.shape)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `make_seq`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the attention weights of a randomly-initialised
> Bahdanau attention as a heatmap over the (target_position, source_position)
> grid. With training, the diagonal would dominate for a reversal task.
> 
> **Why it matters:** attention heatmaps are the standard way to *see* what
> an attention model is doing. They're the deep-learning equivalent of
> feature importance.
> 
> **What to watch for:** with random weights, the heatmap is roughly uniform.
> After training for the reversal task, you'd see a strong anti-diagonal —
> the model learns to attend to the mirror-position input.


### Code cell #5
```python
import matplotlib.pyplot as plt

# Random encoder hidden states for 5 source positions.
h_enc_demo = torch.randn(1, 5, 8)
attn_matrix = []
for t in range(5):
    s = torch.randn(1, 8)
    _, w = attn(h_enc_demo, s)
    attn_matrix.append(w.detach().numpy()[0])
attn_matrix = np.array(attn_matrix)

plt.figure(figsize=(6, 5))
plt.imshow(attn_matrix, cmap="viridis", aspect="auto")
plt.colorbar(label="attention weight")
plt.xlabel("source position"); plt.ylabel("target position")
plt.title("Attention heatmap (untrained, random weights)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Attention is hard to evaluate in isolation — its value comes from improving
> the downstream task. Three common evaluations:
> 
> - **Does attention improve task accuracy?** The real test.
> - **Does the attention heatmap make sense?** Sanity check, not a metric.
> - **Attention rollout / attention flow** — for interpretability research.

> **What this does:** Demonstrates the *qualitative* evaluation: we manually
> set attention weights to perfect anti-diagonal (what a trained reversal
> model should learn) and visualise it. This shows the *target* attention
> pattern.
> 
> **Why it matters:** this is the kind of "did the model learn the right
> thing?" check that distinguishes good NLP engineering from black-box
> modelling.
> 
> **What to watch for:** real attention is rarely this clean. Look for the
> *dominant* pattern, not perfect alignment.


### Code cell #6
```python
# Perfect anti-diagonal: target t attends to source (T - 1 - t).
T = 5
ideal = np.eye(T)[::-1]   # reverse of identity

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(attn_matrix, cmap="viridis", aspect="auto")
axes[0].set_title("untrained attention (random init)")
axes[0].set_xlabel("source position"); axes[0].set_ylabel("target position")
axes[1].imshow(ideal, cmap="viridis", aspect="auto")
axes[1].set_title("ideal attention for sequence reversal")
axes[1].set_xlabel("source position"); axes[1].set_ylabel("target position")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `T = 5`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add attention to the LSTM classifier from notebook 3 — does accuracy improve?
> 2. Train the seq2seq reversal model for 1000 steps and check the attention pattern.
> 3. Replace Bahdanau attention with Luong (dot-product) — which is faster?
> 4. Try *scaled* dot-product (divide by sqrt(d)) — does it help?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a full encoder-decoder LSTM with Bahdanau attention for sequence reversal.
> 2. Implement Luong "general" and "concat" attention variants and compare.
> 3. Implement multi-head attention from scratch (preview of Module 09).
> 4. Build a translation model (English → Pig Latin) with attention.
> 5. Implement attention rollout to aggregate attention across layers.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What problem does attention solve in encoder-decoder models?
> 2. What is the difference between additive (Bahdanau) and multiplicative (Luong) attention?
> 3. Why is scaled dot-product attention preferred in transformers?
> 4. What is the role of the `v` vector in Bahdanau attention?
> 5. How would you visualise what a model attends to?
> 6. Can attention weights be interpreted as feature importance? Why or why not?
> 7. What is multi-head attention, and why use multiple heads?
> 8. How does attention enable parallel training of transformers while RNNs can't?
