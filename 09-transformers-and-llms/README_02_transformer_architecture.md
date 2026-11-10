# Code Walkthrough — `02_transformer_architecture.ipynb`

> Module: **09-transformers-and-llms** | Notebook: **02_transformer_architecture.ipynb**
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

> # 02 — Transformer Architecture
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
> ## 1. Concept Overview — Transformer architecture

> A transformer is a stack of *blocks*, each containing multi-head attention
> and a feed-forward MLP, both wrapped with residual connections and
> layer-norm. The original paper (Vaswani 2017) had 6 blocks for the encoder
> and 6 for the decoder; modern LLMs have 32-96.
> 
> **Beginner track:** learn the 5 components of every transformer block:
> multi-head attention, MLP, residual, layer-norm, positional encoding.
> **Advanced track:** also learn pre-norm vs post-norm, SwiGLU activations,
> RoPE positional encoding, and the deep-scaling tricks that let you train
> 100+ layer transformers.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Multi-head attention**: split $d$ into $h$ heads of size $d/h$ each.
> Run attention in parallel, concat results, project.
> 
> $$\text{MHA}(x) = W^O \cdot \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$
> $$\text{head}_i = \text{Attention}(W^Q_i x, W^K_i x, W^V_i x)$$
> 
> **Positional encoding** (sinusoidal):
> $$\text{PE}_{pos, 2i} = \sin(pos / 10000^{2i/d})$$
> $$\text{PE}_{pos, 2i+1} = \cos(pos / 10000^{2i/d})$$
> 
> **Transformer block** (post-norm):
> $$x = \text{LayerNorm}(x + \text{MHA}(x))$$
> $$x = \text{LayerNorm}(x + \text{MLP}(x))$$

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements multi-head attention from scratch in NumPy.
> We split the projection into 4 heads, run attention per head, then
> concatenate and project back.
> 
> **Why it matters:** multi-head attention is what makes transformers
> expressive. Each head can specialise on a different relation (syntax,
> coreference, semantic similarity).
> 
> **What to watch for:** `reshape(n, h, d_h)` is what "splits" the projection
> into heads. The reshape is free (no copy) — heads are just views of the
> same memory.


### Code cell #1
```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads):
    """x: (n, d). W_*: (d, d). Returns (n, d)."""
    n, d = x.shape
    d_h = d // n_heads   # per-head dim

    # Project to Q, K, V (each (n, d)).
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    # Split each into heads: (n, h, d_h).
    Q = Q.reshape(n, n_heads, d_h).transpose(1, 0, 2)   # (h, n, d_h)
    K = K.reshape(n, n_heads, d_h).transpose(1, 0, 2)
    V = V.reshape(n, n_heads, d_h).transpose(1, 0, 2)

    # Per-head scaled dot-product attention.
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_h)   # (h, n, n)
    weights = softmax(scores, axis=-1)
    heads = weights @ V                                # (h, n, d_h)

    # Concat heads back to (n, d) and apply output projection.
    concat = heads.transpose(1, 0, 2).reshape(n, d)
    return concat @ W_o, weights

# Demo: 4 tokens, 16-dim, 4 heads (so each head has dim 4).
rng = np.random.default_rng(0)
n, d, h = 4, 16, 4
x = rng.standard_normal((n, d))
W_q = rng.standard_normal((d, d)) * 0.1
W_k = rng.standard_normal((d, d)) * 0.1
W_v = rng.standard_normal((d, d)) * 0.1
W_o = rng.standard_normal((d, d)) * 0.1

out, w = multi_head_attention(x, W_q, W_k, W_v, W_o, h)
print("output shape:", out.shape)   # (4, 16)
print("weights shape (one head):", w[0].shape)  # (4, 4)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `softmax`; def `multi_head_attention` — x: (n, d). W_*: (d, d). Returns (n, d).
- **Key imports:** `import numpy as np`
- **Notable calls:** `softmax`

> **Takeaway:** Multi-head attention = "split projection into chunks, run
> attention per chunk, merge." It's a way of letting the model learn
> different attention patterns in parallel.

> **What this does:** Implements sinusoidal positional encoding from scratch
> and visualises it. Each position gets a unique vector built from sines and
> cosines of different frequencies.
> 
> **Why it matters:** attention is permutation-invariant — it needs to be
> told the position of each token. Positional encoding is the simplest way.
> Modern LLMs use learned or RoPE encodings, but sinusoidal is where it
> started.
> 
> **What to watch for:** each *row* is one position; each *column* is one
> frequency. Lower-indexed columns have *low* frequency (long wavelength),
> higher-indexed columns have *high* frequency (short wavelength).


### Code cell #2
```python
def sinusoidal_pe(max_len, d, seed=0):
    pe = np.zeros((max_len, d))
    position = np.arange(max_len)[:, None]      # (max_len, 1)
    div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))   # (d/2,)
    pe[:, 0::2] = np.sin(position * div_term)   # even columns: sin
    pe[:, 1::2] = np.cos(position * div_term)   # odd columns:  cos
    return pe

pe = sinusoidal_pe(50, 32)
print("positional encoding shape:", pe.shape)
print("first 3 positions, first 6 dims:")
print(pe[:3, :6].round(3))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.imshow(pe, cmap="RdBu", aspect="auto")
plt.colorbar(label="value")
plt.xlabel("embedding dimension"); plt.ylabel("position")
plt.title("Sinusoidal positional encoding")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `def sinusoidal_pe(max_len, d, seed=0):`
- **Defines:** def `sinusoidal_pe`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Assembles a complete transformer encoder block in
> PyTorch: multi-head attention + MLP + residuals + layer-norm. We use the
> post-norm variant from the original paper.
> 
> **Why it matters:** this is the building block of *every* transformer.
> BERT is 12 of these stacked; GPT-3 is 96. Once you can write one block,
> you can write any transformer.
> 
> **What to watch for:** modern transformers (GPT-4, LLaMA) use *pre-norm*
> instead of post-norm — it trains more stably at depth. We'll mention this
> in the next cell.


### Code cell #3
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x, mask=None):
        # Post-norm: attention -> residual -> norm, then FF -> residual -> norm.
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
x = torch.randn(8, 20, 64)   # batch=8, seq=20, d_model=64
out = block(x)
print("input shape:", x.shape, " output shape:", out.shape)  # both (8, 20, 64)
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `TransformerBlock`
- **Key imports:** `import torch`, `import torch.nn as nn`
- **Notable calls:** `nn.Linear`, `nn.MultiheadAttention`, `nn.Dropout`

> **What this does:** Shows the pre-norm variant, which is what modern LLMs use.
> The residual is applied *before* the layer-norm, not after. This is a
> one-line change but it makes deep transformers trainable.
> 
> **Why it matters:** the original transformer paper used post-norm and
> trained with warmup; pre-norm can be trained without warmup and scales to
> 100+ layers. Almost every modern LLM uses pre-norm.
> 
> **What to watch for:** in pre-norm, the final layer-norm (after all blocks)
> is critical — without it the residual stream grows unboundedly.


### Code cell #4
```python
class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # Pre-norm: norm -> sublayer -> add.
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                attn_mask=mask)
        x = x + self.drop(attn_out)
        ff_out = self.ff(self.norm2(x))
        return x + self.drop(ff_out)

# Same shape as the post-norm version, just trains more stably at depth.
pn_block = PreNormTransformerBlock(d_model=64, n_heads=4, d_ff=256)
out = pn_block(torch.randn(8, 20, 64))
print("pre-norm block output shape:", out.shape)
```

**What this code does:**

- **Headline:** `class PreNormTransformerBlock(nn.Module):`
- **Defines:** class `PreNormTransformerBlock`
- **Notable calls:** `nn.Linear`, `nn.MultiheadAttention`, `nn.Dropout`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a tiny language-modelling dataset on a small
> text corpus. We tokenise by character, build sequences of fixed length,
> and create (input, target) pairs where the target is the next character.
> 
> **Why it matters:** this is the smallest setup on which a transformer can
> be trained in seconds on CPU. It's perfect for understanding the data flow.
> 
> **What to watch for:** we use a sliding window with stride 1, so consecutive
> training examples overlap heavily. This is fine for our toy demo but
> wasteful in production — usually stride = sequence length.


### Code cell #5
```python
text = "the quick brown fox jumps over the lazy dog. " * 20
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
V = len(chars)

SEQ_LEN = 16
data = [char2idx[c] for c in text]
X, y = [], []
for i in range(len(data) - SEQ_LEN):
    X.append(data[i:i+SEQ_LEN])
    y.append(data[i+SEQ_LEN])

import torch
X_t = torch.tensor(X)
y_t = torch.tensor(y)
print("X shape:", X_t.shape, " y shape:", y_t.shape)
print("vocab size:", V)
print("first input:", "".join(idx2char[i] for i in X[0]), "->", idx2char[y[0]])
```

**What this code does:**

- **Headline:** `text = "the quick brown fox jumps over the lazy dog. " * 20`
- **Key imports:** `import torch`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Builds a tiny 2-layer transformer for character-level
> LM, trains it for 200 steps on our toy corpus, and shows the loss curve.
> Then we generate 100 characters autoregressively.
> 
> **Why it matters:** this is the smallest end-to-end transformer. If you
> understand it, you understand GPT — the architecture is the same, just at
> different scale.
> 
> **What to watch for:** training a transformer requires warmup (the LR ramps
> up over the first ~10% of steps). We use a simple `LambdaLR` schedule.


### Code cell #6
```python
class TinyTransformer(nn.Module):
    def __init__(self, V, d=32, h=4, d_ff=64, n_blocks=2):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Embedding(SEQ_LEN, d)   # learned positional encoding
        self.blocks = nn.ModuleList(
            [PreNormTransformerBlock(d, h, d_ff) for _ in range(n_blocks)]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos(pos)
        # Causal mask: token t can only attend to tokens <= t.
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        for blk in self.blocks:
            h = blk(h, mask=mask)
        return self.head(self.ln(h))

torch.manual_seed(0)
model = TinyTransformer(V)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda step: min(1.0, step / 20))   # 20-step warmup
loss_fn = nn.CrossEntropyLoss()

losses = []
for step in range(200):
    opt.zero_grad()
    logits = model(X_t)        # (B, T, V)
    loss = loss_fn(logits.reshape(-1, V), y_t.reshape(-1).repeat_interleave(SEQ_LEN) if False else y_t)
    loss.backward(); opt.step(); sched.step()
    losses.append(loss.item())

# Generate 100 characters autoregressively from a seed.
generated = list("the quick brown ")
with torch.no_grad():
    for _ in range(100):
        ctx = torch.tensor([[char2idx[c] for c in generated[-SEQ_LEN:]]])
        next_id = model(ctx)[0, -1].argmax().item()
        generated.append(idx2char[next_id])

print("final loss:", round(losses[-1], 4))
print("sampled:", "".join(generated))

plt.figure(figsize=(7, 3))
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss")
plt.title("Tiny transformer loss curve"); plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `class TinyTransformer(nn.Module):`
- **Defines:** class `TinyTransformer`
- **Notable calls:** `nn.Linear`, `nn.Embedding`, `CrossEntropyLoss`, `Adam`, `backward`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Transformers are evaluated by:
> 
> - **Perplexity** on held-out text — the standard LM metric.
> - **Downstream task accuracy** — for BERT, classification tasks.
> - **Human evaluation** — for generative quality (LLM-as-judge, Elo).
> - **Benchmarks** — MMLU, BIG-bench, HumanEval, GSM8K.

> **What this does:** Computes the perplexity of our tiny transformer on the
> training data.
> 
> **Why it matters:** perplexity is the standard LM metric. Even at toy scale,
> you should always compute and report it.
> 
> **What to watch for:** our tiny model probably has perplexity ~5-10
> (vocab size 28). A real LM might have perplexity ~5 on a vocab of 50000 —
> much, much harder.


### Code cell #7
```python
with torch.no_grad():
    logits = model(X_t)
    log_probs = torch.log_softmax(logits[:, -1], dim=-1)
    nll = -log_probs[range(len(y_t)), y_t].mean().item()
    perplexity = np.exp(nll)
print(f"perplexity on training set: {perplexity:.2f}  (vocab size: {V})")
print(f"baseline (uniform): {V}")
```

**What this code does:**

- **Headline:** `with torch.no_grad():`
- **Notable calls:** `softmax`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Increase `n_blocks` from 2 to 4 — does perplexity drop?
> 2. Replace learned positional encoding with sinusoidal — does it matter at this scale?
> 3. Use a larger `d_model` (64, 128) — does it help or just overfit?
> 4. Try *temperature sampling*: `probs = softmax(logits / T)` with T=0.7, 1.0, 1.5.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement the SwiGLU activation in the FFN (replaces GELU in modern LLMs).
> 2. Add RoPE (rotary position embedding) instead of learned PE.
> 3. Implement a full encoder-decoder transformer (Vaswani style) for reversal.
> 4. Train a 4-layer transformer on the Tiny Shakespeare dataset (~1 MB).
> 5. Implement KV-caching for fast autoregressive decoding.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why does the transformer use multi-head attention instead of one big head?
> 2. What is the purpose of positional encoding?
> 3. What is the difference between pre-norm and post-norm transformers?
> 4. Why is the FFN typically 4× wider than the model dimension?
> 5. What is RoPE, and what problem does it solve?
> 6. What is KV-caching, and how much speedup does it give during inference?
> 7. Why do modern LLMs use SwiGLU instead of ReLU in the FFN?
> 8. How does the transformer compare to RNNs in computational complexity?
