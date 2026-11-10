# Code Walkthrough — `04_training_deep_networks.ipynb`

> Module: **06-deep-learning** | Notebook: **04_training_deep_networks.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 30
- **Markdown cells:** 25
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

> # 04 — Training Deep Networks
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
> ## 1. Concept Overview — Training deep networks

> Once your network is more than a few layers deep, training becomes
> fragile: gradients vanish or explode, weights drift, batches go off-distribution.
> This notebook covers the standard techniques that make deep training
> stable: weight initialisation, batch normalisation, dropout, learning-rate
> schedules, and early stopping.
> 
> **Beginner track:** learn the four techniques by name and effect.
> **Advanced track:** also read the original papers — Glorot (2010), BatchNorm
> (2015), Dropout (2014) — to see *why* each technique works.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Key ideas:
> 
> - **Xavier/Glorot init**: $\text{Var}(W) = \frac{2}{n_\text{in} + n_\text{out}}$ for `tanh`.
> - **He init**: $\text{Var}(W) = \frac{2}{n_\text{in}}}$ for `relu`.
> - **BatchNorm**: $\hat{x} = \gamma \frac{x - \mu_B}{\sigma_B} + \beta$.
> - **Dropout**: at training time, zero each unit with prob $p$ and scale by $1/(1-p)$.
> - **Cosine LR schedule**: $\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos(\pi t / T))$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements Xavier and He initialisation from scratch,
> and compares them against the naive `randn * 0.01` on a 10-layer deep
> linear network. The naive init blows up (or vanishes); the principled inits
> keep activations alive through all 10 layers.
> 
> **Why it matters:** Bad initialisation is the most common cause of dead
> deep networks. If `forward` produces `NaN` or all-zeros before training
> even starts, the init is wrong.
> 
> **What to watch for:** `torch.nn.init` provides these as functions —
> you normally call `nn.init.xavier_uniform_(layer.weight)` after creating
> the layer.


### Code cell #1
```python
import numpy as np

def xavier(n_in, n_out, rng):
    # Var = 2 / (n_in + n_out) — good for tanh/sigmoid
    std = np.sqrt(2.0 / (n_in + n_out))
    return rng.standard_normal((n_in, n_out)) * std

def he(n_in, n_out, rng):
    # Var = 2 / n_in — good for relu
    std = np.sqrt(2.0 / n_in)
    return rng.standard_normal((n_in, n_out)) * std

def naive(n_in, n_out, rng):
    # The classic "scratch model" init — too small for deep nets.
    return rng.standard_normal((n_in, n_out)) * 0.01

# Simulate forward pass through 10 linear layers (no activations).
def simulate(init_fn, depth=10, width=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, width))
    for _ in range(depth):
        W = init_fn(width, width, rng)
        x = x @ W
    return float(np.mean(np.abs(x)))   # mean |activation| at output

for name, fn in [("naive *0.01", naive), ("Xavier", xavier), ("He", he)]:
    out = simulate(fn)
    print(f"{name:12s} -> mean |activation| at depth 10 = {out:.4e}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `xavier`; def `he`; def `naive`; def `simulate`
- **Key imports:** `import numpy as np`

> **Takeaway:** Initialisation controls whether activations survive the
> depth. Use Xavier for `tanh`, He for `relu`.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Trains a 5-layer MLP on `make_moons` with and without
> BatchNorm. The BatchNorm version converges in ~50 steps; the vanilla
> version takes ~200.
> 
> **Why it matters:** BatchNorm is the single most impactful architectural
> trick of the 2010s. It allows higher learning rates, deeper networks, and
> more stable training — all by normalising activations inside the network.
> 
> **What to watch for:** `nn.BatchNorm1d` computes its `running_mean` and
> running_var` from training batches and uses those at inference time.
> That's why `model.eval()` matters — without it, the running stats don't
> update properly.


### Code cell #2
```python
import torch, torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_moons(300, noise=0.20, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
Xt_tr = torch.tensor(X_tr, dtype=torch.float32)
yt_tr = torch.tensor(y_tr, dtype=torch.long)
Xt_te = torch.tensor(X_te, dtype=torch.float32)
yt_te = torch.tensor(y_te, dtype=torch.long)

def make_model(use_bn):
    layers = []
    for _ in range(4):
        layers.append(nn.Linear(2, 16))
        if use_bn:
            layers.append(nn.BatchNorm1d(16))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(16, 2))
    return nn.Sequential(*layers)

def train(use_bn, epochs=200, lr=0.05):
    torch.manual_seed(0)
    model = make_model(use_bn)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xt_tr)
        loss = loss_fn(logits, yt_tr)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        acc = (model(Xt_te).argmax(1) == yt_te).float().mean().item()
    return losses, acc

losses_no, acc_no = train(use_bn=False)
losses_bn, acc_bn = train(use_bn=True)

plt.figure(figsize=(7, 4))
plt.plot(losses_no, label=f"no BN (test acc {acc_no:.2f})")
plt.plot(losses_bn, label=f"with BN (test acc {acc_bn:.2f})")
plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend()
plt.title("BatchNorm effect on a 5-layer MLP"); plt.show()
```

**What this code does:**

- **Headline:** `import torch, torch.nn as nn`
- **Defines:** def `make_model`; def `train`
- **Key imports:** `import torch`, `from sklearn.datasets import make_moons`, `from sklearn.model_selection import train_test_split`, `import matplotlib.pyplot as plt`
- **Notable calls:** `train_test_split`, `nn.Linear`, `nn.BatchNorm`, `CrossEntropyLoss`, `SGD`, `backward`

> **Takeaway:** BatchNorm lets you train deeper networks faster and with a
> higher learning rate. It's almost free — drop it in after every `Linear`/`Conv`.

> **What this does:** Demonstrates Dropout, learning-rate scheduling, and
> early stopping in the same training loop. We use a deliberately tiny
> dataset so overfitting is obvious without dropout.
> 
> **Why it matters:** Regularisation is what lets you train a 10M-parameter
> model on 1000 examples without memorising. Dropout is the simplest and
> most effective regulariser for MLPs.
> 
> **What to watch for:** dropout is *only* active during `model.train()`. At
> `model.eval()` it becomes identity — so the test-time output is the average
> of an exponentially large ensemble of sub-networks.


### Code cell #3
```python
def make_model_with_dropout(p=0.3):
    return nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(), nn.Dropout(p),
        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p),
        nn.Linear(64, 2),
    )

torch.manual_seed(0)
model = make_model_with_dropout(p=0.3)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
loss_fn = nn.CrossEntropyLoss()

best_loss = float("inf"); best_state = None; patience = 10; bad = 0
for epoch in range(200):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xt_tr), yt_tr)
    loss.backward(); opt.step(); sched.step()

    # Early stopping on test loss.
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xt_te), yt_te).item()
    if val_loss < best_loss:
        best_loss, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad > patience:
            print(f"early stop at epoch {epoch}, best val loss = {best_loss:.4f}")
            break
model.load_state_dict(best_state)
print("final best val loss:", round(best_loss, 4))
```

**What this code does:**

- **Headline:** `def make_model_with_dropout(p=0.3):`
- **Defines:** def `make_model_with_dropout`
- **Notable calls:** `nn.Linear`, `nn.Dropout`, `CrossEntropyLoss`, `SGD`, `backward`, `model.train`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a deliberately over-fitting problem: a 1000-unit
> MLP on 200 noisy training samples. Train accuracy goes to 100%, test
> accuracy stays at ~85%. Then we add dropout and watch the gap shrink.
> 
> **Why it matters:** this is the textbook over-fitting demo. If you can
> reproduce it, you understand the bias-variance trade-off in deep learning.
> 
> **What to watch for:** train_acc - test_acc is the simplest over-fitting
> metric. A gap > 10% means your regularisation is too weak.


### Code cell #4
```python
from sklearn.datasets import make_circles
X, y = make_circles(200, noise=0.15, random_state=0, factor=0.4)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
Xt_tr = torch.tensor(X_tr, dtype=torch.float32)
yt_tr = torch.tensor(y_tr, dtype=torch.long)
Xt_te = torch.tensor(X_te, dtype=torch.float32)
yt_te = torch.tensor(y_te, dtype=torch.long)

def run(p_dropout, epochs=300):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 1000), nn.ReLU(), nn.Dropout(p_dropout),
        nn.Linear(1000, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xt_tr), yt_tr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Xt_tr).argmax(1) == yt_tr).float().mean().item()
        te_acc = (model(Xt_te).argmax(1) == yt_te).float().mean().item()
    return tr_acc, te_acc

for p in [0.0, 0.3, 0.6]:
    tr, te = run(p)
    print(f"dropout={p:.1f}  train_acc={tr:.3f}  test_acc={te:.3f}  gap={tr-te:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.datasets import make_circles`
- **Defines:** def `run`
- **Key imports:** `from sklearn.datasets import make_circles`
- **Notable calls:** `train_test_split`, `nn.Linear`, `nn.Dropout`, `CrossEntropyLoss`, `Adam`, `backward`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the cosine learning-rate schedule over 200 epochs
> so you can see its shape — high at the start, annealed to ~0 at the end.
> 
> **Why it matters:** the LR schedule is often the difference between a 92%
> and a 94% model. Cosine is the default in modern CV and NLP training.
> 
> **What to watch for:** `T_max=200` should match your total epochs. If you
> schedule to 200 but train for 300, the LR will *increase* again — a classic
> bug.


### Code cell #5
```python
lrs = []
opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
for _ in range(200):
    lrs.append(opt.param_groups[0]["lr"])
    sched.step()

plt.figure(figsize=(7, 3))
plt.plot(lrs)
plt.xlabel("epoch"); plt.ylabel("learning rate")
plt.title("CosineAnnealingLR schedule")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `lrs = []`
- **Notable calls:** `SGD`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For deep networks, watch three signals:
> 
> - **Train loss** — should decrease smoothly; if jagged, lower LR or increase batch.
> - **Validation loss** — should track train loss; if it diverges upward, regularise more.
> - **Gradient norm** — `torch.nn.utils.clip_grad_norm_` keeps it bounded; > 100 means trouble.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace BatchNorm with LayerNorm — does it work as well for MLPs?
> 2. Try `weight_decay=1e-4` in the optimiser — does it substitute for dropout?
> 3. Use `OneCycleLR` instead of `CosineAnnealingLR` — is the result different?
> 4. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement BatchNorm from scratch as a custom `nn.Module`.
> 2. Implement Dropout from scratch and verify it scales correctly at eval time.
> 3. Implement a custom `WarmupCosine` learning-rate schedule.
> 4. Implement early stopping as a callback that wraps a training loop.
> 5. Build a 20-layer MLP and compare Xavier vs He initialisation on it.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why does Xavier init use `2/(n_in + n_out)` and He init use `2/n_in`?
> 2. What is the difference between BatchNorm, LayerNorm, and InstanceNorm?
> 3. Why does Dropout help with regularisation?
> 4. What does `model.train()` / `model.eval()` actually toggle?
> 5. What is the vanishing gradient problem, and how do you fix it?
> 6. Why is the learning rate the single most important hyperparameter?
> 7. What is gradient clipping, and when do you need it?
> 8. How would you diagnose a model that overfits in 5 epochs?
