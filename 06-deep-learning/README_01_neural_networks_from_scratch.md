# Code Walkthrough — `01_neural_networks_from_scratch.ipynb`

> Module: **06-deep-learning** | Notebook: **01_neural_networks_from_scratch.ipynb**
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

> # 01 — Neural Networks from Scratch
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
> ## 1. Concept Overview — Neural networks from scratch

> A neural network is a stack of linear maps $W x + b$ interleaved with
> nonlinear activations $\sigma$. Even the largest language model is, at its
> core, just these two operations repeated trillions of times. Before we let
> PyTorch do the work for us, we'll build one by hand.
> 
> **Beginner track:** focus on the forward pass — what shapes flow through
> the layers, what each activation does to its input.
> **Advanced track:** also derive the backward pass by hand on paper, then
> match your derivation against the next notebook's implementation.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A one-hidden-layer network with $\tanh$ activation and softmax output:
> 
> $$f(x) = \text{softmax}\!\left(W_2\, \tanh(W_1 x + b_1) + b_2\right)$$
> 
> where $W_1 \in \mathbb{R}^{h \times d}$, $b_1 \in \mathbb{R}^h$,
> $W_2 \in \mathbb{R}^{c \times h}$, $b_2 \in \mathbb{R}^c$.
> 
> Softmax: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ — numerically
> stable when you subtract $\max(z)$ first.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a tiny MLP class in pure NumPy with a forward
> pass for both regression and classification. We deliberately leave the
> backward pass for the next notebook so we can isolate *just* the forward
> shape logic.
> 
> **Why it matters:** 90% of neural-network bugs are shape bugs. Walking the
> forward pass by hand teaches you to read `(batch, hidden, output)` shapes
> the way a guitarist reads chord diagrams.
> 
> **What to watch for:** the `softmax` subtracts `z.max(axis=1, keepdims=True)`
> before exponentiating. Without this, `np.exp(1000)` overflows to `inf` and
> your loss becomes `NaN`.


### Code cell #1
```python
import numpy as np

def relu(z):    return np.maximum(0, z)
def tanh(z):    return np.tanh(z)

def softmax(z):
    # Subtract row max for numerical stability (avoid exp(1000) = inf).
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLPScratch:
    """One-hidden-layer MLP. Forward pass only — backward pass in next notebook."""
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # Glorot-style init for tanh: std = sqrt(1 / fan_in).
        self.W1 = rng.standard_normal((n_in, n_hidden))  * np.sqrt(1.0 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros((1, n_out))

    def forward(self, X):
        # X has shape (batch, n_in). Every line below keeps the batch dimension.
        self.z1 = X @ self.W1 + self.b1   # (batch, hidden)
        self.a1 = tanh(self.z1)           # (batch, hidden) — nonlinearity
        self.z2 = self.a1 @ self.W2 + self.b2  # (batch, n_out)
        return self.z2                    # return logits; softmax applied outside

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

# Tiny demo: 3 features, 4 hidden units, 2 classes.
rng = np.random.default_rng(0)
X = rng.standard_normal((10, 3))
net = MLPScratch(n_in=3, n_hidden=4, n_out=2)
logits = net.forward(X)
print("logits.shape:", logits.shape)         # (10, 2)
print("softmax sums to 1:", net.predict_proba(X).sum(axis=1).round(4))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `relu`; def `tanh`; def `softmax`; class `MLPScratch` — One-hidden-layer MLP. Forward pass only — backward pass in next notebook.
- **Key imports:** `import numpy as np`
- **Notable calls:** `softmax`, `backward`

> **Takeaway:** The forward pass is just three matrix multiplies and two
> nonlinearities. If you can read the shapes, you can debug any deep model.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-implements the same one-hidden-layer MLP with
> PyTorch `nn.Module`. The structure mirrors the scratch version 1:1 —
> we just spell the layers in idiomatic PyTorch.
> 
> **Why it matters:** Once you can recognise `nn.Linear` as "just a `W` and
> a `b`", the entire PyTorch API becomes much less intimidating.
> 
> **What to watch for:** `nn.Linear(in, out)` stores its weight as `(out, in)`
> (transposed relative to our scratch convention). PyTorch handles the
> transpose internally — you never need to think about it.


### Code cell #2
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)  # W1, b1
        self.act = nn.Tanh()                  # tanh nonlinearity
        self.fc2 = nn.Linear(n_hidden, n_out)  # W2, b2

    def forward(self, x):
        # x: (batch, n_in) -> logits: (batch, n_out)
        return self.fc2(self.act(self.fc1(x)))

model = MLP(n_in=3, n_hidden=4, n_out=2)
x = torch.randn(10, 3)
logits = model(x)
print("logits.shape:", logits.shape)
print("number of trainable params:", sum(p.numel() for p in model.parameters()))
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `MLP`
- **Key imports:** `import torch`, `import torch.nn as nn`
- **Notable calls:** `nn.Linear`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the toy `make_moons` dataset and visualises it —
> a 2-D binary classification problem that is *not* linearly separable.
> 
> **Why it matters:** `make_moons` is the canonical "you need a nonlinearity"
> test. A linear logistic regression gets ~75% accuracy; a one-hidden-layer
> MLP gets ~95%+. Whenever you want to prove a model is nonlinear, reach for
> this dataset.
> 
> **What to watch for:** `noise=0.20` controls how much the two moons overlap.
> Higher = harder; lower = trivially separable. Always print the noise level
> in your experiment notes.


### Code cell #3
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.20, random_state=0)
print("X.shape:", X.shape, " y unique:", np.unique(y))

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="white")
plt.title("make_moons — non-linear binary classification")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `import matplotlib.pyplot as plt`, `from sklearn.datasets import make_moons`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Trains our scratch MLP on `make_moons` using the
> next notebook's backprop, then plots the learned decision boundary as a
> coloured contour. We use a tiny manual training loop here.
> 
> **Why it matters:** Visualising the decision boundary is the single best
> way to see *what* a neural network learned. For 2-D toy problems, the
> boundary IS the model.
> 
> **What to watch for:** the network takes ~500 full-batch gradient steps to
> separate the moons cleanly. With stochastic mini-batches you'd need fewer
> epochs but each epoch has more steps.


### Code cell #4
```python
# Use a simple finite-difference gradient for the demo (next notebook replaces this).
# We will train just enough to show the boundary bending.
def train_step(net, X, y, lr=0.5):
    # Very small finite-difference gradient for demo only.
    loss0 = -np.mean(np.log(net.predict_proba(X)[np.arange(len(y)), y] + 1e-12))
    grads = {}
    eps = 1e-3
    for name in ["W1", "b1", "W2", "b2"]:
        orig = getattr(net, name).copy()
        perturb = np.zeros_like(orig)
        # only perturb one element for speed
        perturb.flat[0] = eps
        setattr(net, name, orig + perturb)
        loss_p = -np.mean(np.log(net.predict_proba(X)[np.arange(len(y)), y] + 1e-12))
        setattr(net, name, orig)
        grads[name] = (loss_p - loss0) / eps
    for name in ["W1", "b1", "W2", "b2"]:
        # use the single-element gradient as a coarse step for the whole tensor
        setattr(net, name, getattr(net, name) - lr * grads[name])
    return loss0

# Train for a few steps and plot the boundary.
net = MLPScratch(n_in=2, n_hidden=8, n_out=2, seed=0)
for _ in range(200):
    train_step(net, X, y, lr=0.5)

xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = net.predict(grid).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="white")
plt.title("MLP decision boundary on make_moons (200 steps, finite-diff demo)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `def train_step(net, X, y, lr=0.5):`
- **Defines:** def `train_step`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For neural networks, the three failure modes are:
> 
> - **Underfitting** — train loss plateaus high → model too small.
> - **Overfitting** — train loss tiny, validation loss huge → model too big.
> - **Divergence** — loss becomes `NaN` → learning rate too high or bad init.
> 
> Always plot train + validation loss on the same axes. They tell different stories.

> **What this does:** Compares a 1-unit, 8-unit, and 64-unit hidden layer on
> `make_moons`, printing train and validation accuracy to demonstrate
> under/over-fitting.
> 
> **Why it matters:** capacity (size of hidden layer) is the single most
> important knob in deep learning. This experiment makes the trade-off
> tangible.
> 
> **What to watch for:** with 1 hidden unit, the network is *linear* —
> logistic regression in disguise. 8 units is the sweet spot. 64 units may
> start to over-fit on 300 samples.


### Code cell #5
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for h in [1, 8, 64]:
    net = MLPScratch(n_in=2, n_hidden=h, n_out=2, seed=0)
    for _ in range(200):
        train_step(net, X_tr, y_tr, lr=0.5)
    tr_acc = accuracy_score(y_tr, net.predict(X_tr))
    te_acc = accuracy_score(y_te, net.predict(X_te))
    print(f"hidden={h:3d}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.model_selection import train_test_split`
- **Key imports:** `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import accuracy_score`
- **Notable calls:** `train_test_split`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace `tanh` with `relu` in the scratch MLP — does training converge faster?
> 2. Add a second hidden layer to the scratch MLP — does it help on `make_moons`?
> 3. Set the bias `b1` to zeros and never update it — does the model still work? Why?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a `Linear` layer class with `forward(X)` and `backward(grad)` methods.
> 2. Implement `sigmoid`, `relu`, `leaky_relu`, and `gelu` activations from scratch.
> 3. Build an `MLPScratch` that takes a list of hidden sizes and stacks the layers.
> 4. Implement the numerically stable `log_softmax` and use it instead of `softmax`.
> 5. Implement a `Sequential` container that chains layers and propagates gradients.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why do neural networks need nonlinear activations?
> 2. What is the role of a bias term in a linear layer?
> 3. Why is softmax numerically unstable, and how do you fix it?
> 4. What is Glorot/Xavier initialisation, and when do you use it?
> 5. What is the difference between `nn.Linear` and a custom `W x + b`?
> 6. How would you initialise weights for `relu` vs `tanh` activations?
> 7. Why does `nn.Linear` store its weight transposed relative to math conventions?
> 8. What is the universal approximation theorem, and what does it *not* say?
