# Code Walkthrough — `03_rnn_lstm.ipynb`

> Module: **08-natural-language-processing** | Notebook: **03_rnn_lstm.ipynb**
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

> # 03 — RNN & LSTM
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
> ## 1. Concept Overview — Recurrent neural networks and LSTM

> Recurrent neural networks (RNNs) process sequences by maintaining a hidden
> state that gets updated at each step. LSTMs add gates that solve the
> vanishing-gradient problem, allowing the network to remember dependencies
> across hundreds of time steps.
> 
> **Beginner track:** focus on the basic RNN cell and what "hidden state"
> means. LSTMs become natural once you understand RNNs.
> **Advanced track:** also learn GRUs, bidirectional RNNs, and why attention
> replaced RNNs in modern NLP.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Vanilla RNN**: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$.
> 
> **LSTM** with forget, input, output gates:
> $$f_t = \sigma(W_f [h_{t-1}, x_t])$$
> $$i_t = \sigma(W_i [h_{t-1}, x_t])$$
> $$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t])$$
> $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
> $$h_t = o_t \odot \tanh(C_t)$$
> 
> The cell state $C_t$ is the "conveyor belt" — gradients flow through it unchanged.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a vanilla RNN cell and unrolls it for a
> 5-step sequence in pure NumPy. We deliberately use a tiny example so you
> can read every line.
> 
> **Why it matters:** the unrolled RNN is the *archetype* of every sequence
> model — including transformers, which just replace the recurrence with
> attention.
> 
> **What to watch for:** the hidden state is initialised to zeros and gets
> updated at every step. If you reset it inside the loop, the RNN becomes a
> single-step MLP.


### Code cell #1
```python
import numpy as np

class RNNCellScratch:
    def __init__(self, n_in, n_hidden, seed=0):
        rng = np.random.default_rng(seed)
        # Glorot-style init.
        s = np.sqrt(1.0 / (n_in + n_hidden))
        self.W_x = rng.standard_normal((n_in, n_hidden)) * s
        self.W_h = rng.standard_normal((n_hidden, n_hidden)) * s
        self.b   = np.zeros(n_hidden)
    def step(self, x_t, h_prev):
        # x_t: (n_in,)  h_prev: (n_hidden,)
        return np.tanh(x_t @ self.W_x + h_prev @ self.W_h + self.b)

# Unroll over a 5-step sequence.
n_in, n_hidden = 4, 8
cell = RNNCellScratch(n_in, n_hidden, seed=0)

# Random sequence: shape (5, 4) — 5 time steps, 4 features each.
rng = np.random.default_rng(0)
X_seq = rng.standard_normal((5, n_in))

h = np.zeros(n_hidden)
hidden_states = []
for t in range(5):
    h = cell.step(X_seq[t], h)
    hidden_states.append(h.copy())

print("final hidden state (first 4 dims):", h[:4].round(3))
print("hidden state norm at each step:", [round(np.linalg.norm(h), 3) for h in hidden_states])
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** class `RNNCellScratch`
- **Key imports:** `import numpy as np`

> **What this does:** Implements an LSTM cell from scratch in NumPy. It's
> the same shape as the RNN cell, just with 4 gates and a cell state.
> 
> **Why it matters:** the LSTM is the most influential non-transformer
> sequence model. Understanding its gates explains *why* it works on long
> sequences where vanilla RNNs fail.
> 
> **What to watch for:** the `concat` of `[h_prev, x_t]` is then split into
> 4 gates by the weight matrix. This is the "4x the work" pattern every
> LSTM/GRU uses.


### Code cell #2
```python
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

class LSTMCellScratch:
    def __init__(self, n_in, n_hidden, seed=0):
        rng = np.random.default_rng(seed)
        # 4 gates: forget, input, candidate, output — each with its own weight row.
        s = np.sqrt(1.0 / (n_in + n_hidden))
        # Stack all four gate weights into a single matrix for efficiency.
        self.W = rng.standard_normal((n_in + n_hidden, 4 * n_hidden)) * s
        self.b = np.zeros(4 * n_hidden)
        self.n_hidden = n_hidden
    def step(self, x_t, h_prev, c_prev):
        concat = np.concatenate([h_prev, x_t])      # (n_in + n_hidden,)
        gates = concat @ self.W + self.b            # (4 * n_hidden,)
        f, i, g, o = np.split(gates, 4)             # each (n_hidden,)
        f = sigmoid(f); i = sigmoid(i); g = np.tanh(g); o = sigmoid(o)
        c = f * c_prev + i * g                       # cell state update
        h = o * np.tanh(c)                           # hidden state
        return h, c

# Run the LSTM over a 10-step sequence.
n_in, n_hidden = 4, 8
lstm = LSTMCellScratch(n_in, n_hidden, seed=0)
rng = np.random.default_rng(0)
X_seq = rng.standard_normal((10, n_in))

h = np.zeros(n_hidden); c = np.zeros(n_hidden)
for t in range(10):
    h, c = lstm.step(X_seq[t], h, c)
print("LSTM final hidden (first 4 dims):", h[:4].round(3))
print("LSTM cell state (first 4 dims):", c[:4].round(3))
```

**What this code does:**

- **Headline:** `def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))`
- **Defines:** def `sigmoid`; class `LSTMCellScratch`

> **Takeaway:** An LSTM is just an RNN with a "conveyor belt" cell state.
> The gates decide what to keep, what to add, and what to output.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds an LSTM-based sentiment classifier with PyTorch:
> embedding layer → LSTM → dense head. This was the standard NLP architecture
> from ~2015 to ~2018, before transformers took over.
> 
> **Why it matters:** LSTMs are still useful for small-data NLP where
> transformers overfit. The PyTorch code also teaches the `pad_packed_sequence`
> pattern that handles variable-length batches.
> 
> **What to watch for:** `nn.LSTM` returns `(output, (h_n, c_n))`. `h_n`
> is the last hidden state from the *last layer* — that's what you feed to
> the classifier.


### Code cell #3
```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head  = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs.
        e = self.embed(x)              # (batch, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(e) # h_n: (1, batch, hidden_dim)
        return self.head(h_n.squeeze(0))

model = LSTMClassifier(vocab_size=10000, embed_dim=50, hidden_dim=64, n_classes=2)
x = torch.randint(0, 10000, (32, 50))    # batch of 32, each 50 tokens
print("output shape:", model(x).shape)  # (32, 2)
print("parameter count:", sum(p.numel() for p in model.parameters()))
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `LSTMClassifier`
- **Key imports:** `import torch`, `import torch.nn as nn`
- **Notable calls:** `nn.Linear`, `nn.LSTM`, `nn.Embedding`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a tiny synthetic "predict the next character"
> dataset from the string `"hello world"`. The model learns the sequence
> structure of the string.
> 
> **Why it matters:** character-level LMs are the simplest possible
> autoregressive model. If your LSTM can learn `"hello"` you understand the
> essence of language modelling.
> 
> **What to watch for:** with only 11 characters in the corpus, the model
> will overfit immediately. That's fine — we're testing the *architecture*,
> not generalisation.


### Code cell #4
```python
text = "hello world this is a tiny character model"
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
V = len(chars)
print("vocab:", chars, "size:", V)

# Build training pairs: (prev 5 chars) -> next char.
SEQ_LEN = 5
X = []
y = []
for i in range(len(text) - SEQ_LEN):
    X.append([char2idx[c] for c in text[i:i+SEQ_LEN]])
    y.append(char2idx[text[i+SEQ_LEN]])
X = torch.tensor(X)
y = torch.tensor(y)
print("X shape:", X.shape, " y shape:", y.shape)
print("first 3 input/output pairs:")
for i in range(3):
    print(f"  {repr(text[:5])} -> {repr(text[5])}")
```

**What this code does:**

- **Headline:** `text = "hello world this is a tiny character model"`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Trains a tiny LSTM on the character-level task and plots
> the loss curve. We also generate 20 characters by sampling from the model
> autoregressively.
> 
> **Why it matters:** autoregressive generation is how every language model
> produces text — GPT included. Watching an LSTM learn `"hello world"`
> is the same algorithm at tiny scale.
> 
> **What to watch for:** `torch.argmax(logits, dim=-1)` is the *greedy*
> decoding strategy. Real LMs sample from the softmax distribution for
> diversity.


### Code cell #5
```python
class CharLSTM(nn.Module):
    def __init__(self, V, embed=8, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(V, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)
    def forward(self, x):
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.head(out[:, -1])   # only last step

torch.manual_seed(0)
model = CharLSTM(V)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
losses = []
for epoch in range(100):
    opt.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward(); opt.step()
    losses.append(loss.item())

# Generate 20 chars autoregressively.
generated = list("hello")
with torch.no_grad():
    for _ in range(20):
        ctx = torch.tensor([[char2idx[c] for c in generated[-5:]]])
        next_id = model(ctx).argmax(-1).item()
        generated.append(idx2char[next_id])
print("loss after 100 epochs:", round(losses[-1], 4))
print("sampled text:", "".join(generated))

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 3))
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("Char-LSTM loss curve")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `class CharLSTM(nn.Module):`
- **Defines:** class `CharLSTM`
- **Key imports:** `import matplotlib.pyplot as plt`
- **Notable calls:** `nn.Linear`, `nn.LSTM`, `nn.Embedding`, `CrossEntropyLoss`, `Adam`, `backward`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Sequence models are evaluated by:
> 
> - **Perplexity** — $\exp(\text{avg negative log-likelihood})$. Lower is better.
> - **Downstream accuracy** — for classification, standard accuracy/F1.
> - **BLEU / ROUGE** — for translation / summarisation.

> **What this does:** Computes the perplexity of our character LSTM on the
> training data.
> 
> **Why it matters:** perplexity is the standard LM metric. A perplexity of
> $k$ means the model is as uncertain as if choosing uniformly among $k$
> options at each step.
> 
> **What to watch for:** a perfect model has perplexity 1; a model that
> predicts uniformly over $V$ has perplexity $V$.


### Code cell #6
```python
with torch.no_grad():
    logits = model(X)
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs[range(len(y)), y].mean().item()
    perplexity = np.exp(nll)
print(f"final loss: {nll:.4f}")
print(f"perplexity: {perplexity:.3f}  (vocab size: {V})")
print(f"(lower is better; uniform baseline: {V})")
```

**What this code does:**

- **Headline:** `with torch.no_grad():`
- **Notable calls:** `softmax`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace LSTM with GRU — does perplexity change?
> 2. Add a second LSTM layer (`num_layers=2`).
> 3. Make the model bidirectional — does it help for next-char prediction? (Why not?)
> 4. Sample with temperature: `probs = softmax(logits / T)`. Try T=0.5, 1.0, 2.0.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement backprop-through-time (BPTT) for the vanilla RNN from scratch.
> 2. Implement a GRU from scratch and compare with the LSTM.
> 3. Train an LSTM on the Shakespeare subset and generate 1000 characters.
> 4. Implement a bidirectional LSTM for sentiment classification.
> 5. Add attention (Bahdanau) on top of the LSTM and compare.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why do vanilla RNNs suffer from vanishing gradients?
> 2. What do the three LSTM gates do, and why is each one useful?
> 3. What is the cell state in an LSTM, and how does it differ from the hidden state?
> 4. What is backprop-through-time, and why is it expensive?
> 5. What is the difference between LSTM and GRU?
> 6. When would you use a bidirectional LSTM?
> 7. What is perplexity, and what is its interpretation?
> 8. Why did transformers replace LSTMs in modern NLP?
