# Code Walkthrough — `03_optimizers.ipynb`

> Module: **02-machine-learning-fundamentals** | Notebook: **03_optimizers.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 28
- **Markdown cells:** 25
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

> # 03 — Optimizers
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
> ## 1. Concept Overview — Optimizers — how we update the parameters

> Given a loss $L(\theta)$, an **optimizer** is the algorithm that decides the
> next $\theta_{t+1}$ from the current $\theta_t$ and the history of gradients.
> For deep learning and large-scale ML, the choice of optimizer can dominate
> everything else.
> 
> **Beginner track:** focus on the GD vs Momentum vs Adam scratch cell — run
> the trajectory plot and watch the three paths converge.
> **Advanced track:** add Nesterov and RMSProp from scratch, then re-create
> the ill-conditioned experiment to see why Adam beats plain GD on bad
> condition numbers.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> We minimize $L(\theta)$ where $\theta \in \mathbb{R}^d$.
> 
> - **GD**: $\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta_t)$
> - **Momentum**: $v_{t+1} = \mu v_t + \nabla L$; $\theta_{t+1} = \theta_t - \eta v_{t+1}$
> - **Nesterov**: evaluate gradient at the *look-ahead* point.
> - **AdaGrad**: scale each coordinate by the sqrt of squared-gradient history.
> - **RMSProp**: AdaGrad with exponential decay of the gradient history.
> - **Adam**: momentum + RMSProp + bias correction.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements vanilla Gradient Descent, Momentum, and Adam
> from scratch on a 2-D quadratic loss $L(\theta) = \tfrac{1}{2}\theta^\top A
> \theta - b^\top \theta$, then runs all three from the same starting point
> and prints where each lands.
> 
> **Why it matters:** These are the three optimizers behind every neural
> network ever trained. Once you can write each in 10 lines, you understand
> what `torch.optim.Adam` is doing under the hood — and why its learning-rate
> schedule matters.
> 
> **What to watch for:** Adam's bias-correction terms `1 - b1**t` and
> `1 - b2**t` look like bookkeeping, but they are *critical* in the first ~50
> steps — without them, the moving averages start at 0 and the first update
> is far too large.


### Code cell #1
```python
import numpy as np

def quadratic(theta, A, b):
    # L(theta) = 0.5 theta^T A theta - b^T theta ; grad = A theta - b
    return 0.5 * theta @ A @ theta - b @ theta, A @ theta - b

def gd(grad, theta0, lr=0.05, steps=200):
    theta = theta0.copy()                       # never mutate theta0
    path = [theta.copy()]                       # record trajectory for plotting
    for _ in range(steps):
        theta = theta - lr * grad(theta)        # plain gradient step
        path.append(theta.copy())
    return theta, np.array(path)

def momentum(grad, theta0, lr=0.05, mu=0.9, steps=200):
    theta = theta0.copy()
    v = np.zeros_like(theta)                    # velocity (Polyak heavy-ball)
    path = [theta.copy()]
    for _ in range(steps):
        v = mu * v + grad(theta)                 # accumulate direction
        theta = theta - lr * v                  # step in accumulated direction
        path.append(theta.copy())
    return theta, np.array(path)

def adam(grad, theta0, lr=0.05, b1=0.9, b2=0.999, eps=1e-8, steps=200):
    theta = theta0.copy()
    m = np.zeros_like(theta); v = np.zeros_like(theta)   # 1st, 2nd moments
    path = [theta.copy()]
    for t in range(1, steps+1):
        g = grad(theta)
        m = b1 * m + (1 - b1) * g                # 1st moment (mean)
        v = b2 * v + (1 - b2) * g * g            # 2nd moment (uncentered var)
        m_hat = m / (1 - b1 ** t)                # bias-corrected 1st moment
        v_hat = v / (1 - b2 ** t)                # bias-corrected 2nd moment
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)   # RMSProp-scaled step
        path.append(theta.copy())
    return theta, np.array(path)

A = np.array([[3.0, 0.2], [0.2, 1.0]])           # SPD quadratic bowl
b = np.array([1.0, 2.0])
grad_fn = lambda th: A @ th - b                  # gradient of the quadratic
theta_star = np.linalg.solve(A, b)               # closed-form minimum
print("closed-form minimum:", theta_star)

theta0 = np.array([4.0, -4.0])                   # starting point for all 3
_, p_gd  = gd(grad_fn, theta0, lr=0.05, steps=200)
_, p_mo  = momentum(grad_fn, theta0, lr=0.05, mu=0.9, steps=200)
_, p_ad  = adam(grad_fn, theta0, lr=0.1, steps=200)
print("GD final:        ", p_gd[-1])
print("Momentum final:   ", p_mo[-1])
print("Adam final:       ", p_ad[-1])
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `quadratic`; def `gd`; def `momentum`; def `adam`
- **Key imports:** `import numpy as np`

> **Takeaway:** GD crawls down the bowl axis-by-axis; Momentum surges through
> the long axis and damps the short one; Adam adapts its step size per
> coordinate and converges in the fewest steps. Same loss, very different
> trajectories — and that is *the* reason deep learning uses Adam.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-runs Adam on the same quadratic, this time using
> PyTorch's `torch.optim.Adam` and autograd — so we get the gradient for free
> from `loss.backward()` instead of coding it by hand.
> 
> **Why it matters:** This is the *exact* code pattern you will use to train
> every neural network. Compare it to the scratch Adam — only the gradient
> computation changes; the optimizer logic is identical.
> 
> **What to watch for:** `opt.zero_grad()` *must* be called before each
> `loss.backward()` — otherwise gradients accumulate across steps and the
> effective learning rate grows without bound.


### Code cell #2
```python
# PyTorch optimizers on the same quadratic problem.
import torch

A_t = torch.tensor(A); b_t = torch.tensor(b)               # NumPy -> torch
theta = torch.tensor([4.0, -4.0], requires_grad=True)     # leaf we optimise
opt = torch.optim.Adam([theta], lr=0.1)                    # Adam on theta

for step in range(200):
    opt.zero_grad()                       # clear last step gradients
    loss = 0.5 * theta @ A_t @ theta - b_t @ theta
    loss.backward()                       # autograd fills theta.grad
    opt.step()                            # Adam update using theta.grad
print("PyTorch Adam final:", theta.detach().numpy())
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`

> **Takeaway:** PyTorch's `torch.optim.Adam` is doing *exactly* what our
> scratch Adam did — `b1`, `b2`, `eps`, and bias correction are all the same
> constants. The only difference is autograd: you write the loss, PyTorch
> computes the gradient.

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> Optimizers live on the loss *landscape*. For a logistic regression on the
> `breast_cancer` dataset, the loss is convex — every reasonable optimizer
> converges to the same point; only the *path* differs.

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Overlays the three optimizer trajectories (GD, Momentum,
> Adam) on a contour plot of the quadratic loss surface. Each curve is the
> path the optimizer took from `(4, -4)` to the minimum.
> 
> **Why it matters:** This is the visual that makes optimizers click. GD
> zig-zags across the bowl (because the gradient is dominated by the steep
> axis); Momentum smooths the zig-zag; Adam takes a near-straight line.
> 
> **What to watch for:** the contour grid is built with `np.meshgrid` and the
> loss is evaluated *manually* per grid point — a list comprehension over
> 40000 points. Vectorise this in your own code with broadcasting.


### Code cell #3
```python
import matplotlib.pyplot as plt

# Contour plot of the quadratic loss with the three paths overlaid.
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
Z = np.array([0.5 * np.array([x, y]) @ A @ np.array([x, y]) - b @ np.array([x, y])
              for x, y in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(xx, yy, Z, levels=20, cmap="viridis")       # loss contours
ax.plot(*theta_star, "k*", markersize=15, label="minimum")  # true minimum
ax.plot(p_gd[:, 0], p_gd[:, 1],  label="GD",      lw=2)
ax.plot(p_mo[:, 0], p_mo[:, 1],  label="Momentum", lw=2)
ax.plot(p_ad[:, 0], p_ad[:, 1],  label="Adam",     lw=2)
ax.set_title("Optimizer trajectories on a quadratic bowl")
ax.legend(); plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> **Takeaway:** When the loss surface is an elongated bowl (the typical case
> for ill-conditioned problems), plain GD wastes most of its steps oscillating
> across the short axis. Momentum and Adam fix this — Momentum by averaging,
> Adam by per-coordinate scaling.

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> We compare optimizers by:
> 
> - **Convergence speed** — how many steps to reach a given loss?
> - **Final loss** — do they actually reach the optimum?
> - **Stability** — do they oscillate or diverge for large learning rates?

> ---
> 
> ## 8. Experiments — Experiments

> 1. Make `A` badly conditioned — e.g. `A = diag([100, 1])` — and watch plain GD
>    oscillate. Does Adam fix it?
> 2. Vary the learning rate of Momentum from 0.01 to 0.5 — what happens at the
>    stability limit $\eta = 2/\lambda_{\max}$?
> 3. Replace the loss with a non-convex function (e.g. Rosenbrock) and compare.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement Nesterov accelerated gradient and compare it visually to Momentum.
> 2. Implement AdaGrad and RMSProp from scratch and contrast on the ill-conditioned quadratic.
> 3. Add learning-rate *warmup* to Adam and show that it stabilises the first 50 steps.
> 4. Implement a *line search* — at each step, find the optimal step length.
> 5. Build a cosine-annealing learning-rate schedule.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What does the learning rate *actually* control, and what happens when it is too large or too small?
> 2. Why does momentum help in ravines (long narrow valleys)?
> 3. Why is Adam often preferred over plain SGD for deep learning?
> 4. What is the bias-correction step in Adam and why does it matter?
> 5. When does AdaGrad fail, and how does RMSProp fix it?
> 6. What is the difference between convex and non-convex optimisation for ML?
> 7. How would you choose an optimizer for a 1-billion-parameter language model?
> 8. What is the relationship between SGD with momentum and Polyak heavy-ball method?
