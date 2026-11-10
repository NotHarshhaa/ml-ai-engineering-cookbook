# Code Walkthrough — `04_gpt_clm.ipynb`

> Module: **09-transformers-and-llms** | Notebook: **04_gpt_clm.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 30
- **Markdown cells:** 24
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

> # 04 — GPT & Causal Language Modeling
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
> ## 1. Concept Overview — GPT and causal language modelling

> GPT (Generative Pre-trained Transformer, Radford 2018) is the canonical
> *decoder* transformer. It's pretrained by predicting the next token in a
> sequence (the *CLM* objective). The same architecture scales to GPT-3,
> GPT-4, LLaMA, and beyond.
> 
> **Beginner track:** understand the CLM objective and why the causal mask
> is essential.
> **Advanced track:** also learn KV-caching, beam search, top-k/top-p
> sampling, and the scaling laws that predict loss from compute.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Causal LM objective**: maximise
> $$L = \frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$
> 
> with $P(x_t \mid x_{<t}) = \text{softmax}(W^O h_t)$ and $h_t$ from a
> causally-masked transformer.
> 
> **Sampling** at generation time: choose $x_t \sim P(\cdot \mid x_{<t})$.
> Variants: greedy (argmax), beam search, top-k, top-p (nucleus), temperature.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements the causal-LM loss in NumPy — cross-entropy
> over the next-token predictions at every position. We use the same `softmax`
> from earlier notebooks.
> 
> **Why it matters:** the CLM loss is the *only* training signal for GPT.
> Once you can compute it by hand, you understand exactly what GPT is being
> trained to do.
> 
> **What to watch for:** the loss shifts the predictions by one position —
> logits at position $t$ predict token $t+1$. Off-by-one here is a classic
> bug that produces a model that predicts the *current* token perfectly and
> the next token terribly.


### Code cell #1
```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def causal_lm_loss(logits, targets):
    """logits: (T, V), targets: (T,) — targets[t] is the token at position t+1."""
    log_probs = np.log(softmax(logits, axis=-1) + 1e-12)   # (T, V)
    nll = -log_probs[np.arange(len(targets)), targets]    # (T,)
    return nll.mean()

# Tiny demo: 5 tokens, vocab size 10.
rng = np.random.default_rng(0)
T, V = 5, 10
logits = rng.standard_normal((T, V))
input_ids = rng.integers(0, V, T)
# Targets are shifted by 1: target[t] = input_ids[t+1].
targets = np.roll(input_ids, -1)
print("input :", input_ids)
print("targets:", targets)
print("CLM loss:", causal_lm_loss(logits, targets).round(4))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `softmax`; def `causal_lm_loss` — logits: (T, V), targets: (T,) — targets[t] is the token at position t+1.
- **Key imports:** `import numpy as np`
- **Notable calls:** `softmax`

> **What this does:** Implements top-k and top-p (nucleus) sampling from a
> next-token distribution. Both are the standard ways to sample from GPT at
> inference time.
> 
> **Why it matters:** pure greedy decoding (argmax) is repetitive; pure random
> sampling is incoherent. Top-k and top-p strike the balance that makes GPT
> output readable.
> 
> **What to watch for:** `top_p=0.9` keeps the smallest set of tokens whose
> cumulative probability is ≥ 0.9. `top_k=40` keeps the 40 most-likely tokens
> regardless of probability mass.


### Code cell #2
```python
def top_k_sample(probs, k=40, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    probs = probs ** (1.0 / temperature)
    probs = probs / probs.sum()
    # Keep only the top-k tokens; zero out the rest.
    top_idx = np.argsort(probs)[-k:]
    masked = np.zeros_like(probs)
    masked[top_idx] = probs[top_idx]
    masked = masked / masked.sum()
    return rng.choice(len(probs), p=masked)

def top_p_sample(probs, p=0.9, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    probs = probs ** (1.0 / temperature)
    probs = probs / probs.sum()
    # Sort descending, find smallest set whose cumulative prob >= p.
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    cutoff = (cum < p).sum() + 1   # number of tokens to keep
    keep = order[:cutoff]
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    masked = masked / masked.sum()
    return rng.choice(len(probs), p=masked)

# Demo: a hand-crafted next-token distribution.
probs = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.01, 0.01, 0.03])
for _ in range(5):
    print(f"top-k={top_k_sample(probs, k=3, seed=_)}, top-p={top_p_sample(probs, p=0.5, seed=_)}")
```

**What this code does:**

- **Headline:** `def top_k_sample(probs, k=40, temperature=1.0, seed=0):`
- **Defines:** def `top_k_sample`; def `top_p_sample`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Loads `distilgpt2` (the smallest GPT-2 variant) from
> HuggingFace and generates 50 tokens from a prompt. DistilGPT2 has ~82M
> parameters and runs on CPU in seconds.
> 
> **Why it matters:** this is the smallest "real" LLM you can experiment
> with. The patterns (tokenisation, generation loop, sampling) carry over to
> GPT-3, GPT-4, LLaMA, Mistral, etc.
> 
> **What to watch for:** we use `return_dict_in_generate=True` to get the
> full generation metadata, including per-step attention weights if you
> ask for them.


### Code cell #3
```python
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()

    prompt = "Once upon a time, the AI engineer decided"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print("prompt:", prompt)
    print("token count:", input_ids.shape[1])

    # Generate 50 tokens with top-p=0.9 sampling.
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(input_ids, do_sample=True, max_new_tokens=50,
                              top_p=0.9, temperature=0.8)
    print("\nGPT-2 continuation:")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print(f"skipped GPT-2 demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from transformers import GPT2Tokenizer`
- **Notable calls:** `GPT2Tokenizer`, `GPT2LMHeadModel`, `model.generate`, `model.eval`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the tiny Shakespeare subset (~1 MB) and tokenises
> it. This is the classic training data for tiny GPT demos.
> 
> **Why it matters:** Tiny Shakespeare is the *hello world* of language
> modelling. A 6-layer transformer trained for ~10 minutes can produce
> plausible-looking Shakespeare. It's the smallest dataset on which an LM
> produces coherent long-form output.
> 
> **What to watch for:** the entire dataset is ~1 MB. A real LLM is trained
> on ~terabytes of data — 6 orders of magnitude more.


### Code cell #4
```python
try:
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare.txt"
    text = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
    print("dataset size:", len(text), "chars")
    print("first 200 chars:")
    print(text[:200])
except Exception as e:
    print(f"using synthetic corpus ({e})")
    text = "To be or not to be, that is the question. " * 200
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import urllib.request`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the next-token probability distribution for
> a single prompt. We plot the top-15 candidate tokens by probability.
> 
> **Why it matters:** seeing the distribution is the most direct way to
> understand what an LLM is doing. Each step of generation is *exactly* this:
> pick a token from a distribution.
> 
> **What to watch for:** GPT-2's logits are unbounded; for visualisation we
> apply softmax to convert them to probabilities that sum to 1.


### Code cell #5
```python
try:
    prompt = "The most important thing in machine learning is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]   # (vocab,)
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(15)
    tokens = [tokenizer.decode([i]) for i in top.indices]
    values = top.values.numpy()

    plt.figure(figsize=(10, 4))
    plt.bar(tokens, values, color="steelblue", edgecolor="white")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("probability")
    plt.title(f"Next-token distribution after: \"{prompt}\"")
    plt.tight_layout(); plt.show()
except Exception as e:
    print(f"skipped next-token plot ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Notable calls:** `softmax`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> GPT-style models are evaluated by:
> 
> - **Perplexity** on held-out text — direct LM metric.
> - **Downstream benchmarks** — MMLU, BIG-bench, HumanEval, GSM8K, TruthfulQA.
> - **Human evaluation** — chatbot Elo ratings, preference rankings.
> - **LM-as-judge** — use a bigger LM to score outputs of a smaller one.

> **What this does:** Computes the perplexity of `distilgpt2` on a snippet
> of text.
> 
> **Why it matters:** perplexity is the most common LM metric. Lower is
> better; GPT-2 small achieves ~30 on web text, GPT-3 reaches ~15.
> 
> **What to watch for:** we average the per-token NLL, then exponentiate.
> Doing it in this order is correct; doing it in the reverse order is wrong.


### Code cell #6
```python
try:
    sample_text = "To be or not to be, that is the question."
    input_ids = tokenizer.encode(sample_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1]   # predict token t+1 from token t
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).mean().item()
    perplexity = np.exp(nll)
    print(f"perplexity of distilgpt2 on \"{sample_text}\": {perplexity:.2f}")
    print("(vocab size:", tokenizer.vocab_size, ")")
except Exception as e:
    print(f"skipped perplexity demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Notable calls:** `softmax`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Try `top_k=10`, `top_k=50`, `top_k=200` — how does generation quality change?
> 2. Try `temperature=0.3`, `1.0`, `1.5` — what do you notice?
> 3. Use `num_beams=5` for beam search — does it produce better text than sampling?
> 4. Generate 200 tokens from a prompt of your choice and rate the output.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement KV-caching from scratch and benchmark the speedup.
> 2. Implement beam search from scratch (without `model.generate`).
> 3. Train a tiny GPT (6 layers, 128-dim) on Tiny Shakespeare for 30 min.
> 4. Implement speculative decoding with a smaller draft model.
> 5. Implement a chat template and fine-tune a small GPT for conversation.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between BERT's MLM and GPT's CLM objectives?
> 2. Why does GPT use a causal mask?
> 3. What is the difference between top-k and top-p sampling?
> 4. What is temperature, and how does it affect generation?
> 5. What is KV-caching, and how much speedup does it give?
> 6. What is the difference between greedy decoding and beam search?
> 7. What is perplexity, and why is it exponentiated negative log-likelihood?
> 8. What is the difference between pretraining and instruction tuning?
