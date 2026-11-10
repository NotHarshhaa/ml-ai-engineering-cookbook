# Code Walkthrough — `02_backpropagation_from_scratch.ipynb`

> Module: **06-deep-learning** | Notebook: **02_backpropagation_from_scratch.ipynb**
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

> # 02 — Backpropagation from Scratch
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
> ## 1. Concept Overview — Backpropagation from scratch

> Backpropagation is just the chain rule applied to a computation graph.
> Once you've derived the gradients for an MLP by hand, every deep learning
> framework stops being magical — they're all doing the same algorithm.
> 
> **Beginner track:** derive the gradient of a 1-layer network by hand on
> paper, then run this notebook and check your answer.
> **Advanced track:** also derive the gradient for a 2-layer network, then
> verify with `torch.autograd.gradcheck`.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> For the network $f(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$ with loss
> $L = \frac{1}{2} \|f(x) - y\|^2$:
> 
> $$\frac{\partial L}{\partial W_2} = (\hat{y} - y)\, \sigma(z_1)^\top$$
> $$\frac{\partial L}{\partial W_1} = \big( W_2^\top (\hat{y} - y) \big) \odot \sigma'(z_1)\, x^\top$$
> 
> The pattern: **error at the output, propagated backward through the
> weights, modulated by the local activation derivative.**

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a one-hidden-layer MLP with both forward and
> backward passes in pure NumPy. The backward pass returns the gradients of
> all four parameters (`W1, b1, W2, b2`).
> 
> **Why it matters:** This is the smallest interesting example of
> backpropagation — small enough to print every gradient, large enough to
> exercise the chain rule twice.
> 
> **What to watch for:** `tanh_deriv(z) = 1 - tanh(z)**2` is the *local*
> gradient — it tells you how `tanh` responds to small perturbations in
> its input. The backward pass multiplies it by the *upstream* gradient.


### Code cell #1
```python
import numpy as np

def tanh(z):       return np.tanh(z)
def tanh_deriv(z): return 1.0 - np.tanh(z) ** 2   # local gradient

class MLPBackprop:
    """MLP with explicit forward and backward passes."""
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(1.0 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros((1, n_out))

    def forward(self, X):
        # Cache intermediate values — needed by backward().
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.yhat = self.z2  # identity output for regression
        return self.yhat

    def backward(self, dL_dyhat):
        # dL_dyhat has shape (batch, n_out) — upstream gradient.
        # 1. Output layer: linear, so local gradient is 1.
        dL_dz2 = dL_dyhat
        dL_dW2 = self.a1.T @ dL_dz2                     # (hidden, n_out)
        dL_db2 = dL_dz2.sum(axis=0, keepdims=True)      # (1, n_out)
        # 2. Hidden layer: chain rule through tanh.
        dL_da1 = dL_dz2 @ self.W2.T                     # (batch, hidden)
        dL_dz1 = dL_da1 * tanh_deriv(self.z1)            # multiply by local grad
        dL_dW1 = self.X.T @ dL_dz1                      # (n_in, hidden)
        dL_db1 = dL_dz1.sum(axis=0, keepdims=True)      # (1, hidden)
        return {"W1": dL_dW1, "b1": dL_db1, "W2": dL_dW2, "b2": dL_db2}

# Sanity check: fit a tiny regression problem.
rng = np.random.default_rng(0)
X = rng.standard_normal((50, 3))
true_w = np.array([[1.0], [-2.0], [0.5]])
y = np.tanh(X @ true_w) + 0.05 * rng.standard_normal((50, 1))

net = MLPBackprop(3, 8, 1, seed=0)
lr = 0.05
losses = []
for epoch in range(500):
    yhat = net.forward(X)
    loss = 0.5 * np.mean((yhat - y) ** 2)
    losses.append(loss)
    dL_dyhat = (yhat - y) / len(X)   # mean over batch
    grads = net.backward(dL_dyhat)
    for name in ["W1", "b1", "W2", "b2"]:
        setattr(net, name, getattr(net, name) - lr * grads[name])

print(f"initial loss: {losses[0]:.4f}")
print(f"final loss:   {losses[-1]:.4f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `tanh`; def `tanh_deriv` — MLP with explicit forward and backward passes.; class `MLPBackprop` — MLP with explicit forward and backward passes.
- **Key imports:** `import numpy as np`
- **Notable calls:** `backward`

> **Takeaway:** Backpropagation is just `upstream_grad * local_grad` chained
> through the layers. There is no other algorithm — PyTorch, TensorFlow,
> JAX all build a graph that does exactly this.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Trains the same regression problem with PyTorch's
> `autograd` — the framework does backprop automatically from the forward
> graph.
> 
> **Why it matters:** Once you've written backprop by hand, you'll never
> trust a framework that hides it from you. PyTorch's `loss.backward()` is
> doing exactly what your scratch code did — it just walks the computation
> graph automatically.
> 
> **What to watch for:** `optimizer.zero_grad()` before each `backward()` —
> PyTorch *accumulates* gradients by default. Forgetting to zero is the #1
> PyTorch footgun.


### Code cell #2
```python
import torch
import torch.nn as nn

torch.manual_seed(0)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1)
)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for epoch in range(500):
    opt.zero_grad()                       # forget this -> gradients accumulate
    yhat = model(X_t)
    loss = loss_fn(yhat, y_t)
    loss.backward()                        # autograd fills in p.grad for every p
    opt.step()                             # optimizer uses p.grad to update p

print(f"PyTorch final loss: {loss.item():.4f}")
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`, `import torch.nn as nn`
- **Notable calls:** `nn.Linear`, `MSELoss`, `SGD`, `autograd`, `backward`

> **What this does:** Uses `torch.autograd.gradcheck` to numerically verify
> that our hand-derived gradient matches a finite-difference estimate.
> 
> **Why it matters:** `gradcheck` is the gold standard for sanity-checking
> custom autograd functions. If your gradient passes `gradcheck`, it is
> correct (within floating-point tolerance).
> 
> **What to watch for:** `gradcheck` needs `double` precision (float64). The
> default `float32` is too imprecise for finite-difference comparison.


### Code cell #3
```python
# Use gradcheck to verify a custom autograd Function.
from torch.autograd import Function, gradcheck

class TanhWithGrad(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.tanh(x)
        ctx.save_for_backward(x, out)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        x, out = ctx.saved_tensors
        return grad_out * (1 - out ** 2)   # local gradient of tanh

# gradcheck compares autograd grad with finite-difference grad.
x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
ok = gradcheck(lambda inp: TanhWithGrad.apply(inp), (x,), eps=1e-6)
print("gradcheck passed?", ok)
```

**What this code does:**

- **Headline:** `from torch.autograd import Function, gradcheck`
- **Defines:** class `TanhWithGrad`
- **Key imports:** `from torch.autograd import Function`
- **Notable calls:** `autograd`, `backward`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small *non-linear regression* problem: fit
> $y = \sin(x) + 0.1 \epsilon$ using an MLP. We plot data, predictions, and
> the loss curve.
> 
> **Why it matters:** A linear model fundamentally cannot fit $\sin(x)$ —
> this is the cleanest demonstration of *why* you need a nonlinearity.
> 
> **What to watch for:** the network extrapolates *flat* outside the training
> data range. NNs do not extrapolate — they interpolate.


### Code cell #4
```python
rng = np.random.default_rng(0)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(x) + 0.1 * rng.standard_normal((100, 1))

net = MLPBackprop(1, 16, 1, seed=0)
lr = 0.05
for epoch in range(2000):
    yhat = net.forward(x)
    dL_dyhat = (yhat - y) / len(x)
    grads = net.backward(dL_dyhat)
    for name in ["W1", "b1", "W2", "b2"]:
        setattr(net, name, getattr(net, name) - lr * grads[name])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(x, y, s=8, label="data", alpha=0.6)
plt.plot(x, net.forward(x), color="red", label="MLP fit", lw=2)
plt.title("Fitting sin(x) with a 16-unit MLP"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(losses if "losses" in dir() else [])
plt.title("Loss curve"); plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(0)`
- **Notable calls:** `backward`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** A 3-panel diagnostic for the trained regression MLP:
> data + fit, residuals, and loss curve.
> 
> **Why it matters:** the residual plot catches *bias* in the fit (a curve
> in residuals means the model is too small). The loss curve catches
> *instability* (jagged = learning rate too high).
> 
> **What to watch for:** a well-trained MLP on `sin(x)` will have residuals
> that look like pure noise — no curve, no trend.


### Code cell #5
```python
preds = net.forward(x)
resid = y - preds

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(x, y, s=8, alpha=0.6, label="data")
axes[0].plot(x, preds, color="red", lw=2, label="fit")
axes[0].set_title("Fit"); axes[0].legend()
axes[1].scatter(x, resid, s=8, alpha=0.6)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residuals")
# Re-train briefly to capture a clean loss curve.
lc = []
net2 = MLPBackprop(1, 16, 1, seed=0)
for _ in range(500):
    yhat = net2.forward(x)
    lc.append(0.5 * np.mean((yhat - y) ** 2))
    dL = (yhat - y) / len(x)
    g = net2.backward(dL)
    for n in ["W1", "b1", "W2", "b2"]:
        setattr(net2, n, getattr(net2, n) - 0.05 * g[n])
axes[2].plot(lc); axes[2].set_title("Loss curve (first 500 steps)")
axes[2].set_yscale("log")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `preds = net.forward(x)`
- **Notable calls:** `backward`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> The single most important diagnostic for a neural network is the **loss
> curve**. Three things to look for:
> 
> - It should *decrease* — if not, your learning rate is wrong or your loss is wrong.
> - It should *plateau* — if not, you stopped training too early.
> - It should be *smooth* — if jagged, lower the learning rate or increase batch size.

> **What this does:** Trains three networks with learning rates 0.001,
> 0.05, and 1.0 to show the three failure modes: too-small (slow),
> just-right (converges), too-large (diverges).
> 
> **Why it matters:** Picking the learning rate is the single most impactful
> decision in deep learning. This experiment builds intuition for the *sign*
> of "too high" vs "too low".
> 
> **What to watch for:** at `lr=1.0`, the loss may become `NaN`. That's the
> divergence failure mode — gradient steps overshoot the minimum.


### Code cell #6
```python
for lr in [0.001, 0.05, 1.0]:
    net = MLPBackprop(1, 16, 1, seed=0)
    losses = []
    for _ in range(300):
        yhat = net.forward(x)
        losses.append(0.5 * np.mean((yhat - y) ** 2))
        if not np.isfinite(losses[-1]):
            break
        dL = (yhat - y) / len(x)
        g = net.backward(dL)
        for n in ["W1", "b1", "W2", "b2"]:
            setattr(net, n, getattr(net, n) - lr * g[n])
    final = losses[-1] if losses else float("nan")
    status = "diverged" if not np.isfinite(final) else "ok"
    print(f"lr={lr:6.3f}  initial={losses[0]:.4f}  final={final:.4f}  status={status}")
```

**What this code does:**

- **Headline:** `for lr in [0.001, 0.05, 1.0]:`
- **Notable calls:** `backward`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add momentum to the SGD loop — does the loss curve get smoother?
> 2. Replace `tanh` with `relu` — what changes?
> 3. Train the same network with batch size 8 (mini-batch SGD) — is it faster?
> 4. Add L2 weight decay `lr * (W - lam * W)` — does it help on a noisy dataset?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement backprop for a 2-hidden-layer MLP by hand and verify with `gradcheck`.
> 2. Implement backprop for softmax + cross-entropy (combined, simpler than separate).
> 3. Implement a tiny autograd: a `Tensor` class with `.backward()` that walks the graph.
> 4. Add momentum to the scratch optimizer and compare convergence.
> 5. Implement gradient checking for the entire network.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why do we need to cache intermediate values during the forward pass?
> 2. What is the chain rule, in one sentence?
> 3. Why does `tanh` suffer from vanishing gradients while `relu` does not?
> 4. What is the local gradient of softmax with respect to its input?
> 5. Why does PyTorch accumulate gradients by default?
> 6. What is the difference between `loss.backward()` and `torch.autograd.grad`?
> 7. How would you debug a NaN loss in a deep network?
> 8. What is `gradcheck` and when do you use it?
