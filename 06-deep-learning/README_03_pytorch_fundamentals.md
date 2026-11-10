# Code Walkthrough — `03_pytorch_fundamentals.ipynb`

> Module: **06-deep-learning** | Notebook: **03_pytorch_fundamentals.ipynb**
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

> # 03 — PyTorch Fundamentals
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
> ## 1. Concept Overview — PyTorch fundamentals

> PyTorch is the dominant deep-learning framework in research. Three core
> abstractions: **tensors** (n-d arrays with GPU support), **autograd**
> (automatic differentiation through a dynamic graph), and **nn.Module**
> (composable layers).
> 
> **Beginner track:** master tensor creation, indexing, and the training
> loop. Everything else is variation on these.
> **Advanced track:** also learn `torch.compile`, custom `autograd.Function`,
> and distributed training primitives (`DistributedDataParallel`).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> PyTorch doesn't introduce new math — it implements linear algebra on a
> computation graph. The math you need:
> 
> - Tensors: $\mathbf{X} \in \mathbb{R}^{n_1 \times n_2 \times \ldots}$.
> - Operations: matmul, broadcasting, reductions (same as NumPy).
> - Gradients: $\frac{\partial L}{\partial \theta}$ via reverse-mode autodiff.
> - SGD: $\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a tiny autograd from scratch — a `Tensor` class
> that records its parent operation and a `backward()` method that walks the
> graph. This is the *heart* of PyTorch, in 30 lines.
> 
> **Why it matters:** Once you've written this, `loss.backward()` stops being
> magic. You'll see it as a graph traversal that your own code can do.
> 
> **What to watch for:** the backward pass uses **topological sort** — visit
> every node only after all its children have been visited.


### Code cell #1
```python
# A tiny autograd engine — the heart of PyTorch in 30 lines.
class Tensor:
    """Minimal autograd tensor — wraps a numpy array + a graph node."""
    def __init__(self, data, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort of all parents.
        topo = []
        visited = set()
        def build(n):
            if n not in visited:
                visited.add(n)
                for p in n._parents:
                    build(p)
                topo.append(n)
        build(self)
        self.grad = np.ones_like(self.data)  # seed
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

# Verify: f(x, y) = x*y + x, df/dx = y + 1, df/dy = x.
x = Tensor([3.0]); y = Tensor([4.0])
z = x * y + x
z.backward()
print("dz/dx =", x.grad, "(expected 5)")
print("dz/dy =", y.grad, "(expected 3)")
```

**What this code does:**

- **Headline:** `class Tensor:`
- **Defines:** class `Tensor` — Minimal autograd tensor — wraps a numpy array + a graph node.
- **Notable calls:** `autograd`, `backward`

> **Takeaway:** Reverse-mode autodiff is just graph traversal plus the
> chain rule. PyTorch does this for graphs with millions of nodes.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Shows the four everyday PyTorch patterns: tensor
> creation, broadcasting, GPU transfer, and the training loop. Nothing
> fancy — just the muscle memory you need.
> 
> **Why it matters:** 95% of PyTorch bugs come from one of these four areas:
> wrong dtype, wrong device, missing `zero_grad()`, or detached tensors.
> 
> **What to watch for:** `tensor.to(device)` is the canonical way to move
> between CPU and GPU. `tensor.cuda()` works but is less flexible.


### Code cell #2
```python
import torch

# 1. Tensor creation: from Python list, numpy, or random.
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.randn(2, 2)            # standard normal
print("a + b =\n", a + b)

# 2. Broadcasting: same rules as NumPy.
row = torch.tensor([[1.0, 2.0, 3.0]])   # shape (1, 3)
col = torch.tensor([[10.0], [20.0]])   # shape (2, 1)
print("row + col =\n", row + col)        # shape (2, 3)

# 3. Device transfer (no-op if no GPU).
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(100, 10).to(device)
print(f"tensor on device: {x.device}")

# 4. The canonical training loop (just the skeleton).
model = torch.nn.Linear(10, 1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
for step in range(3):
    X = torch.randn(32, 10).to(device)
    y = torch.randn(32, 1).to(device)
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    print(f"step {step}  loss = {loss.item():.4f}")
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`
- **Notable calls:** `nn.Linear`, `MSELoss`, `SGD`, `backward`

> **What this does:** Builds a custom `nn.Module` — the same way PyTorch
> ships `nn.Linear`, `nn.Conv2d`, etc. You write a class with `__init__`
> (declare sub-modules and parameters) and `forward` (the computation).
> 
> **Why it matters:** every model in `torchvision`, `transformers`, etc. is
> an `nn.Module`. Knowing the pattern lets you read and modify any of them.
> 
> **What to watch for:** call `super().__init__()` first, or your sub-modules
> won't be registered and `model.parameters()` will be empty.


### Code cell #3
```python
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()              # critical — registers sub-modules
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)

model = TwoLayerMLP(10, 32, 2)
print("parameter count:", sum(p.numel() for p in model.parameters()))
print("sub-modules:", list(model._modules.keys()))
```

**What this code does:**

- **Headline:** `import torch.nn as nn`
- **Defines:** class `TwoLayerMLP`
- **Key imports:** `import torch.nn as nn`
- **Notable calls:** `nn.Linear`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Wraps a NumPy array in a `torch.utils.data.Dataset` and
> iterates through it with a `DataLoader` — the canonical PyTorch data
> pipeline.
> 
> **Why it matters:** the `Dataset` + `DataLoader` pattern is used by every
> PyTorch training script. It handles batching, shuffling, multi-process
> loading, and collation — all things you don't want to write yourself.
> 
> **What to watch for:** `batch_size=32` is the default for most image and
> tabular work. For LLMs the effective batch size is often much larger
> (via gradient accumulation).


### Code cell #4
```python
from torch.utils.data import Dataset, DataLoader

class SinDataset(Dataset):
    def __init__(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(-3, 3, n).astype(np.float32).reshape(-1, 1)
        self.y = (np.sin(self.x) + 0.1 * rng.standard_normal((n, 1))).astype(np.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

ds = SinDataset(200)
loader = DataLoader(ds, batch_size=32, shuffle=True)

for i, (xb, yb) in enumerate(loader):
    print(f"batch {i}: x shape={xb.shape} y shape={yb.shape}")
    if i == 2: break
```

**What this code does:**

- **Headline:** `from torch.utils.data import Dataset, DataLoader`
- **Defines:** class `SinDataset`
- **Key imports:** `from torch.utils.data import Dataset`
- **Notable calls:** `DataLoader`, `Dataset`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Trains our `TwoLayerMLP` on the `sin(x)` dataset and
> plots the loss curve + final fit.
> 
> **Why it matters:** this is the *smallest* end-to-end PyTorch training
> loop. Once you can read it, you can read any PyTorch training script.
> 
> **What to watch for:** `model.train()` and `model.eval()` switch dropout
> and batchnorm between training and inference modes. Forgetting `eval()`
> during testing is a classic bug.


### Code cell #5
```python
model = TwoLayerMLP(1, 32, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
losses = []

for epoch in range(200):
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses); axes[0].set_title("Loss curve"); axes[0].set_yscale("log")

xs = torch.linspace(-3, 3, 200).reshape(-1, 1)
model.eval()                  # turn off dropout/batchnorm if any
with torch.no_grad():         # don't track gradients during inference
    ys = model(xs).numpy()
axes[1].scatter(ds.x, ds.y, s=8, alpha=0.5, label="data")
axes[1].plot(xs.numpy(), ys, color="red", lw=2, label="fit")
axes[1].set_title("MLP fit"); axes[1].legend()
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `model = TwoLayerMLP(1, 32, 1)`
- **Notable calls:** `MSELoss`, `Adam`, `backward`, `model.eval`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Three things to check in any PyTorch training loop:
> 
> - **Loss decreases smoothly** — if jagged, lower the learning rate or increase batch size.
> - **`model.eval()` for inference** — otherwise dropout and batchnorm behave wrongly.
> - **`torch.no_grad()` for inference** — otherwise you accumulate useless gradients and waste memory.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace `Adam` with `SGD` — does it converge?
> 2. Add `nn.Dropout(0.2)` after the activation — does it slow training?
> 3. Train on GPU (`device='cuda'`) and compare wall-clock time.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a custom `nn.Module` that exposes `forward(x, return_features=True)`.
> 2. Implement a learning-rate scheduler (`StepLR`, `CosineAnnealingLR`).
> 3. Implement gradient accumulation for an effective batch size of 256.
> 4. Build a custom `collate_fn` for a DataLoader with variable-length sequences.
> 5. Profile a training loop with `torch.profiler` and find the bottleneck.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between `torch.tensor` and `np.array`?
> 2. Why does PyTorch use dynamic computation graphs?
> 3. What is the difference between `model.train()` and `model.eval()`?
> 4. When do you need `torch.no_grad()`?
> 5. What is `autograd.Function`, and when would you write a custom one?
> 6. How does `DistributedDataParallel` differ from `DataParallel`?
> 7. What is mixed-precision training, and what are its trade-offs?
> 8. How would you debug an OOM error in a training loop?
