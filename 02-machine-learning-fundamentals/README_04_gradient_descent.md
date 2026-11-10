# Code Walkthrough — `04_gradient_descent.ipynb`

> Module: **02-machine-learning-fundamentals** | Notebook: **04_gradient_descent.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 27
- **Markdown cells:** 24
- **Code cells:** 3

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

> # 04 — Gradient Descent
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
> ## 1. Concept Overview — Gradient Descent — the engine of modern ML

> Gradient Descent (GD) is the *family* of algorithms that includes every
> optimizer in the previous notebook. We devote a notebook to it because it
> is the conceptual hinge between calculus and ML.
> 
> **Beginner track:** implement Batch GD, SGD, and Mini-batch GD on linear
> regression — watch the three loss curves converge at different rates.
> **Advanced track:** add gradient checking (compare analytic vs numerical
> gradient within 1e-6) and a learning-rate schedule — these are the two
> debugging tools you will use in every deep-learning training loop.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Taylor's theorem says $L(\theta + \Delta) \approx L(\theta) + \nabla L(\theta)^\top \Delta$.
> For a step $\Delta = -\eta \nabla L(\theta)$, the loss decreases whenever
> $\eta$ is small enough. Three flavours of GD appear in ML:
> 
> - **Batch GD**: gradient over the full dataset.
> - **Stochastic GD (SGD)**: gradient on one sample at a time.
> - **Mini-batch GD**: gradient on a batch — the practical compromise.
> 
> Convergence rate for convex $L$ is $O(1/\sqrt{t})$ for SGD, $O(1/t)$ for GD.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Sets up a 200×3 linear regression problem with a known
> ground-truth weight vector, then trains three optimizers on it — Batch GD
> (gradient over all 200 samples), SGD (one sample per step), and Mini-batch
> GD (16 samples per step). Each records the training MSE per step.
> 
> **Why it matters:** These are the three flavors of GD that run every ML
> training loop in production. Watching their loss curves side-by-side is
> the cleanest illustration of the *bias of the gradient estimator* vs the
> *variance of the gradient estimator* trade-off.
> 
> **What to watch for:** SGD uses `rng.integers(len(y))` to pick *one* sample
> per step — its loss curve will be *noisy* but descend on average. Mini-batch
> uses `rng.choice(len(y), size=batch, replace=False)` — no replacement within
> a batch, so every batch is a true subset.


### Code cell #1
```python
import numpy as np

rng = np.random.default_rng(0)
# Linear regression: y = X w + noise.   Loss(w) = ||Xw - y||^2 / (2n)
n, d = 200, 3
X = rng.standard_normal((n, d))
w_true = np.array([1.5, -2.0, 0.5])                  # ground-truth weights
y = X @ w_true + 0.1 * rng.standard_normal(n)        # noisy targets

def loss_grad(w, X, y):
    return X.T @ (X @ w - y) / len(y)                  # 1/n * X^T (Xw - y)

def batch_gd(X, y, lr=0.1, steps=200):
    w = np.zeros(X.shape[1])                          # init at zero
    hist = []
    for _ in range(steps):
        g = loss_grad(w, X, y)                        # full-data gradient
        w = w - lr * g
        hist.append(np.mean((X @ w - y) ** 2))       # record full-data MSE
    return w, hist

def sgd(X, y, lr=0.05, steps=200, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(steps):
        i = rng.integers(len(y))                     # one random sample
        xi, yi = X[i:i+1], y[i:i+1]
        g = xi.T @ (xi @ w - yi)        # gradient on one sample
        w = w - lr * g
        hist.append(np.mean((X @ w - y) ** 2))       # eval on full data
    return w, hist

def minibatch_gd(X, y, lr=0.07, batch=16, steps=200, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(steps):
        idx = rng.choice(len(y), size=batch, replace=False)  # 16 distinct rows
        Xb, yb = X[idx], y[idx]
        g = Xb.T @ (Xb @ w - yb) / batch              # mean batch gradient
        w = w - lr * g
        hist.append(np.mean((X @ w - y) ** 2))
    return w, hist

w_bgd, h_bgd = batch_gd(X, y, lr=0.1, steps=200)
w_sgd, h_sgd = sgd(X, y, lr=0.05, steps=200)
w_mgd, h_mgd = minibatch_gd(X, y, lr=0.07, batch=16, steps=200)

print("true w     :", w_true)
print("batch GD   :", w_bgd)
print("SGD        :", w_sgd)
print("mini-batch :", w_mgd)
print("closed form:", np.linalg.lstsq(X, y, rcond=None)[0])
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `loss_grad`; def `batch_gd`; def `sgd`; def `minibatch_gd`
- **Key imports:** `import numpy as np`

> **Takeaway:** Batch GD's loss curve is smooth but each step is expensive;
> SGD's is noisy but each step is O(1) in the dataset size; Mini-batch is
> the practical compromise that scales to billion-row datasets — and it is
> what every deep-learning trainer actually runs.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-runs SGD on the same regression problem using
> `sklearn.linear_model.SGDRegressor`, and compares its weights to the
> closed-form OLS solution.
> 
> **Why it matters:** `SGDRegressor` is the production-grade version of the
> SGD we just wrote. Watching it agree with our scratch code (and with OLS)
> is the cheapest sanity check that our scratch implementation is correct.
> 
> **What to watch for:** `learning_rate="constant"` plus `tol=None` plus
> `max_iter=200` makes the sklearn SGD run for *exactly* 200 iterations —
> otherwise it stops early when the loss stops improving, which would make
> the comparison to our scratch code unfair.


### Code cell #2
```python
from sklearn.linear_model import SGDRegressor, LinearRegression

sgd = SGDRegressor(learning_rate="constant", eta0=0.05, max_iter=200,
                   tol=None, random_state=0).fit(X, y)
print("sklearn SGD weights:", sgd.coef_)
print("sklearn OLS weights:", LinearRegression().fit(X, y).coef_)
```

**What this code does:**

- **Headline:** `from sklearn.linear_model import SGDRegressor, LinearRegression`
- **Key imports:** `from sklearn.linear_model import SGDRegressor`
- **Notable calls:** `LinearRegression`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> For visualisation, we descend a 2-D loss surface — the logistic regression
> loss on two features of the iris dataset — and watch the trajectory.

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the training MSE of all three GD variants on a
> log scale. The y-axis is loss; the x-axis is the step number. Log scale is
> used because the three curves span several orders of magnitude.
> 
> **Why it matters:** This is the canonical *training curve* plot. Batch GD
> is smooth and monotonic; SGD is noisy but descends on average; Mini-batch
> is in between. Reading these curves is the most common debugging step in
> deep-learning training.
> 
> **What to watch for:** `ax.set_yscale("log")` is what makes the early
> loss visible — on a linear scale, the curves all collapse to the bottom
> after step 50 and you cannot see the difference between them.


### Code cell #3
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h_bgd, label=f"Batch GD (final loss {h_bgd[-1]:.4f})")
ax.plot(h_sgd, label=f"SGD      (final loss {h_sgd[-1]:.4f})", alpha=0.8)
ax.plot(h_mgd, label=f"Mini-batch (final loss {h_mgd[-1]:.4f})")
ax.set_xlabel("step"); ax.set_ylabel("MSE")
ax.set_title("Training-loss curves for three GD variants")
ax.set_yscale("log"); ax.legend(); plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> **Takeaway:** Log-scale training curves are the universal language of
> training-loop debugging. If a curve is flat, your learning rate is too
> small; if it diverges, too large; if it zig-zags, your batch size is too
> small. The shape tells you what to fix.

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Convergence of GD is assessed by:
> 
> - **Loss curve** — does it decrease monotonically (batch) or with noise (SGD)?
> - **Distance to optimum** — $\|\theta_t - \theta^*\|$.
> - **Gradient norm** — should approach 0 as we approach a stationary point.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Standardise `X` (subtract mean, divide by std) and re-run GD — does the
>    convergence speed change? Why?
> 2. Add an $L_2$ regularisation term $\lambda \|w\|^2$ to the gradient.
> 3. Compare the *wall-clock* time of 200 steps of Batch GD vs 200 steps of SGD
>    on a 100k-row dataset.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement *gradient checking*: compare your analytic gradient to a
>    finite-difference approximation within $10^{-6}$.
> 2. Implement the **logistic regression** training loop with mini-batch GD.
> 3. Add an early-stopping criterion based on the change in loss.
> 4. Implement a *step-decay* learning-rate schedule and compare to constant LR.
> 5. Implement GD with momentum and verify it accelerates convergence on an
>    ill-conditioned problem.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why is mini-batch GD the dominant choice in deep learning?
> 2. What is the trade-off between batch size and learning rate?
> 3. Why does feature scaling matter for GD?
> 4. What is the difference between a *stationary point* and a *minimum*?
> 5. How would you debug a training loop whose loss diverges?
> 6. Why does SGD converge at all, given the gradient is noisy?
> 7. What is *vanishing gradient* and how does it differ from a bad learning rate?
> 8. Explain the $O(1/\sqrt{t})$ vs $O(1/t)$ convergence rates in plain English.
