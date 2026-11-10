# Code Walkthrough — `02_numpy_fundamentals.ipynb`

> Module: **01-python-for-ml** | Notebook: **02_numpy_fundamentals.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 34
- **Markdown cells:** 26
- **Code cells:** 8

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

> # 02 — NumPy Fundamentals
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
> ## 1. Concept Overview — NumPy — the computational engine of ML

> NumPy provides the n-dimensional array (`ndarray`): a contiguous, typed block
> of memory with vectorised operations. Almost every Python ML framework
> (scikit-learn, PyTorch, TensorFlow, JAX) is built on top of NumPy conventions
> or mirrors them.
> 
> **Why this matters for ML:** vectorisation is the difference between a
> training loop that runs in 3 seconds and one that runs in 3 minutes.
> 
> **Beginner track:** focus on shapes. Print `arr.shape` after every line.
> **Advanced track:** check `arr.dtype` and `arr.flags` — `C_CONTIGUOUS` vs
> `F_CONTIGUOUS` matters when you're feeding data to BLAS.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Key ideas from linear algebra that NumPy exposes directly:
> 
> - Vector $\mathbf{x} \in \mathbb{R}^n$ and matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$.
> - Element-wise operations: $\mathbf{a} \odot \mathbf{b}$.
> - Dot product: $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$.
> - Matrix product: $\mathbf{C} = \mathbf{A}\mathbf{B}$.
> - Broadcasting: aligning shapes so a scalar (or row) combines with a matrix.
> - Norm: $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements dot product and L2 norm in pure Python.
> These are the two operations at the heart of every ML algorithm — distance
> calculations, weight updates, attention scores.
> 
> **Why it matters:** Once you've written `dot` by hand you understand *why*
> `a @ b` is fast in numpy: it's not the math that's slow in Python, it's the
> per-element dispatch. C code does the same math with no interpreter overhead.
> 
> **What to watch for:** `zip(a, b)` stops at the shorter sequence — silently.
> Always check `len(a) == len(b)` first in your own code.


### Code cell #1
```python
import math

def dot_python(a, b):
    """Σ_i a_i * b_i — pure Python."""
    return sum(ai * bi for ai, bi in zip(a, b))

def norm_python(a):
    """||a||_2 = sqrt(a . a)."""
    return math.sqrt(dot_python(a, a))

a = [1.0, 2.0, 3.0, 4.0]
b = [4.0, 3.0, 2.0, 1.0]
print("dot:", dot_python(a, b))      # 1*4 + 2*3 + 3*2 + 4*1 = 20
print("norm of a:", norm_python(a))  # sqrt(1+4+9+16) = sqrt(30) ≈ 5.48
```

**What this code does:**

- **Headline:** `import math`
- **Defines:** def `dot_python` — Σ_i a_i * b_i — pure Python.; def `norm_python` — ||a||_2 = sqrt(a . a).
- **Key imports:** `import math`

> **What this does:** The same two operations with NumPy, plus the *outer*
> product — three lines that demonstrate the matrix-as-block-of-memory
> philosophy.
> 
> **Why it matters:** NumPy delegates the actual arithmetic to highly tuned
> C/Fortran (BLAS). You write Python; BLAS does the math. This is what makes
> ML training loops feasible in Python at all.
> 
> **What to watch for:** the `@` operator and `np.dot` are *not* identical
> for high-dimensional arrays — `@` always does matrix-multiply on the last


### Code cell #2
```python
import numpy as np

a_np = np.array(a, dtype=np.float64)  # explicit dtype avoids surprise upcasts
b_np = np.array(b, dtype=np.float64)
print("dot:", a_np @ b_np)              # @ is matrix-multiply (np.matmul)
print("norm:", np.linalg.norm(a_np))   # L2 by default; pass ord=1 for L1
print("element-wise:", a_np * b_np)     # element-wise (Hadamard) product
print("outer product:\n", np.outer(a_np, b_np))  # shape (4, 4): a_i * b_j
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds a noisy linear-regression dataset `X, y` using
> matrix operations, then explores centring (subtracting the column mean) and
> QR decomposition as a way to produce a (close to) orthogonal matrix.
> 
> **Why it matters:** These are the exact primitives you will use to *train*
> and *evaluate* every linear model. The shape of `X` is `(n_samples,
> n_features)` — never transpose it. `y` is `(n_samples, 1)` so it broadcasts.
> 
> **What to watch for:** broadcasting subtracts `X.mean(axis=0)` (shape
> `(5,)`) from `X` (shape `(1000, 5)`) — NumPy aligns the trailing axes, so
> this *just works*. Always print shapes if you're confused.


### Code cell #3
```python
# The everyday NumPy toolkit.
rng = np.random.default_rng(seed=42)   # the modern, recommended RNG

# Shapes you will see in every ML codebase.
X = rng.standard_normal((1000, 5))   # design matrix: 1000 samples, 5 features
w = rng.standard_normal((5, 1))      # weight vector (column vector)
b = np.array([[0.3]])                # bias, shape (1,1) so it broadcasts
y = X @ w + b + 0.1 * rng.standard_normal((1000, 1))  # noisy targets
print("X.shape:", X.shape, "y.shape:", y.shape)

# Broadcasting: subtract the column mean (centring).
# This is the first step of standardisation and PCA.
X_centered = X - X.mean(axis=0)
print("column means after centring:", X_centered.mean(axis=0).round(4))

# QR decomposition: produces an orthogonal Q and upper-triangular R.
# We use this in linear-algebra-heavy ML to avoid inverting ill-conditioned matrices.
Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
print("Q^T Q close to identity?:\n", (Q.T @ Q).round(3))
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(seed=42)   # the modern, recommended RNG`
- **Notable calls:** `PCA`

> **What this does:** Solves linear regression with the closed-form least
> squares solution $w^* = (X^\top X)^{-1} X^\top y$ using `np.linalg.lstsq`.
> 
> **Why it matters:** `lstsq` is preferred over `inv(X.T @ X) @ X.T @ y`
> because (a) it handles rank-deficient `X` gracefully and (b) it uses a
> numerically stable SVD-based solver under the hood.
> 
> **What to watch for:** we explicitly stack a column of ones onto `X` to
> give the model an intercept term. If you forget this, the regression is
> forced through the origin and your coefficients will look wrong.


### Code cell #4
```python
# Linear-algebra solver: closed-form linear regression.
# w* = (X^T X)^-1 X^T y, but we use lstsq for numerical stability.
Xb = np.hstack([X, np.ones((X.shape[0], 1))])   # add bias column
w_closed, *_ = np.linalg.lstsq(Xb, y, rcond=None)  # least squares
print("closed-form weights (including bias):\n", w_closed.ravel())

# Compare against the true w used to generate the data.
# We won't recover it exactly because of the 0.1*noise we added.
print("true weights (with bias 0.3):", w.ravel(), "[0.3]")
```

**What this code does:**

- **Headline:** `Xb = np.hstack([X, np.ones((X.shape[0], 1))])   # add bias column`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the iris CSV with `np.genfromtxt`, splitting the
> 4 numeric features from the species string column.
> 
> **Why it matters:** NumPy can read CSVs directly, but only for *homogeneous*
> data. Once you have mixed types (numbers + strings), `pandas` becomes the
> right tool. We do it the numpy way once so you appreciate the pandas way.
> 
> **What to watch for:** `dtype=None` tells NumPy to infer per-column dtypes
> — this is the only way to keep the species string column. Without it,
> everything becomes a float and the species column becomes NaN.


### Code cell #5
```python
# Load the iris CSV via NumPy only.
import numpy as np
from urllib.request import urlretrieve
import tempfile, os

csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
urlretrieve(csv_url, tmp.name)

# Skip header; last column is the species string.
# dtype=None + encoding="utf-8" returns a structured array (one type per column).
raw = np.genfromtxt(tmp.name, delimiter=",", skip_header=1, dtype=None,
                    encoding="utf-8")
os.unlink(tmp.name)

# Stack the first 4 columns (the numeric features) into a 2-D float array.
features = np.stack([row[:4] for row in raw]).astype(np.float64)
species = np.array([row[4] for row in raw])
print("feature matrix shape:", features.shape)
print("feature means per column:", features.mean(axis=0).round(3))
print("unique species:", np.unique(species))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `from urllib.request import urlretrieve`, `import tempfile`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Two panels — a histogram of sepal length, and a
> scatter of sepal length vs width colour-coded by species.
> 
> **Why it matters:** these two plots are the *first* thing you draw on any
> new dataset. The histogram tells you the marginal distribution of a
> feature; the scatter tells you whether classes are linearly separable in
> that 2-D slice.
> 
> **What to watch for:** `edgecolor="white"` on `hist` and `scatter` removes
> the muddy edge that makes dense plots unreadable. Small styling choices
> matter when you're presenting to a stakeholder.


### Code cell #6
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: histogram of one feature.
axes[0].hist(features[:, 0], bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Sepal length distribution")
axes[0].set_xlabel("cm"); axes[0].set_ylabel("count")

# Right: scatter of two features, coloured by species.
colors = {"setosa": "tab:red", "versicolor": "tab:green", "virginica": "tab:blue"}
for sp in np.unique(species):
    mask = species == sp
    axes[1].scatter(features[mask, 0], features[mask, 1],
                    label=sp, alpha=0.7, edgecolor="white")
axes[1].set_xlabel("sepal length (cm)"); axes[1].set_ylabel("sepal width (cm)")
axes[1].set_title("Sepal length vs width by species")
axes[1].legend()
plt.tight_layout()  # avoid label overlap
plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> **Takeaway:** Histogram + coloured scatter is your default first look at
> any tabular dataset. It catches scale issues, class structure, and
> outliers in one glance.

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> **What this does:** Computes the $R^2$ (coefficient of determination) for the
> closed-form linear regression we fit earlier on synthetic data.
> 
> **Why it matters:** $R^2 = 1 - SS_\text{res}/SS_\text{tot}$ is the standard
> regression score — 1.0 is a perfect fit, 0.0 means "no better than
> predicting the mean", negative values mean worse than the mean.
> 
> **What to watch for:** a high $R^2$ on the *training* data does not imply
> generalisation — you must compute it on a *test* set or via cross-validation.


### Code cell #7
```python
# Verify our closed-form weights by computing R^2 on the noisy synthetic data.
y_hat = Xb @ w_closed                    # predictions
ss_res = np.sum((y - y_hat) ** 2)        # residual sum of squares
ss_tot = np.sum((y - y.mean()) ** 2)     # total sum of squares (around mean)
r2 = 1 - ss_res / ss_tot                 # coefficient of determination
print(f"R^2 on synthetic regression: {r2.item():.4f}")
```

**What this code does:**

- **Headline:** `y_hat = Xb @ w_closed                    # predictions`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Time `X @ X.T` for `X.shape == (2000, 2000)` against a pure-Python triple loop.
> 2. Replace `np.linalg.lstsq` with the normal equations `np.linalg.inv(X.T @ X) @ X.T @ y` — when does this become numerically unstable?
> 3. Compare memory use of `float64` vs `float32` arrays of shape `(1_000_000, 100)`.

> **What this does:** Times a 2000×2000 matrix multiplication with NumPy, and
> compares the memory footprint of a 1M×100 matrix stored as `float64` vs
> `float32`.
> 
> **Why it matters:** A 2000×2000 matmul finishes in milliseconds in NumPy —
> the same loop in pure Python would take minutes. And `float32` halves your
> memory, which on a GPU is often the difference between fitting and OOM.
> 
> **What to watch for:** `time.perf_counter()` (not `time.time()`) is the
> right timer for short intervals — it's monotonic and high-resolution.


### Code cell #8
```python
import time
big = np.random.default_rng(0).standard_normal((2000, 2000))
t0 = time.perf_counter()
_ = big @ big.T
print(f"2000x2000 matmul: {time.perf_counter()-t0:.3f}s")

# float32 = 4 bytes/element vs float64 = 8 bytes/element.
# On 1M x 100 = 1e8 elements that's 800 MB vs 400 MB.
f64 = np.zeros((1_000_000, 100), dtype=np.float64)
f32 = np.zeros((1_000_000, 100), dtype=np.float32)
print(f"float64 size: {f64.nbytes / 1e6:.1f} MB")
print(f"float32 size: {f32.nbytes / 1e6:.1f} MB  (half the memory)")
```

**What this code does:**

- **Headline:** `import time`
- **Key imports:** `import time`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement `softmax(z)` with NumPy in a numerically stable way (subtract the row max).
> 2. Implement `standardize(X)` that returns `(X - mean) / std` per column.
> 3. Write `knn_predict(X_train, y_train, X_test, k=5)` using only `np.linalg.norm`.
> 4. Implement the Moore-Penrose pseudoinverse of a non-square matrix.
> 5. Build a function that returns the top-k eigenvalues of a covariance matrix.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is broadcasting, and what are the three rules for shape compatibility?
> 2. Why is a vectorised NumPy operation faster than an equivalent Python loop?
> 3. When would you use `np.linalg.lstsq` instead of inverting a matrix?
> 4. What is the difference between `np.matmul`, `@`, and `np.dot`?
> 5. How do you avoid the copy-on-assignment footgun with `arr2 = arr1` vs `arr2 = arr1.copy()`?
> 6. What does `axis=0` vs `axis=1` mean for a 2-D array reduction?
> 7. Explain why `np.random.default_rng(seed)` is preferable to `np.random.seed`.
> 8. How would you compute pairwise Euclidean distances without an explicit loop?
