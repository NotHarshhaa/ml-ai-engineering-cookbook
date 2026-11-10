# Code Walkthrough — `02_loss_functions.ipynb`

> Module: **02-machine-learning-fundamentals** | Notebook: **02_loss_functions.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 33
- **Markdown cells:** 27
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

> # 02 — Loss Functions
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
> ## 1. Concept Overview — Loss functions — what we ask the model to minimise

> The **loss function** $\ell(\hat{y}, y)$ measures the cost of predicting
> $\hat{y}$ when the truth is $y$. Choice of loss *is* choice of inductive
> bias — MSE punishes large errors, MAE punishes all errors linearly, log-loss
> is proper for probabilities, hinge loss is maximum-margin.
> 
> **Beginner track:** focus on the five scratch loss functions — copy each
> into your editor and re-derive the gradient by hand.
> **Advanced track:** add the gradient of each loss and verify numerically
> with finite differences — that is the bridge to Module 03 (Optimizers).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Regression losses**
> - MSE: $\ell(\hat{y}, y) = (\hat{y} - y)^2$
> - MAE: $\ell(\hat{y}, y) = |\hat{y} - y|$
> - Huber: quadratic for $|e| \le \delta$, linear otherwise.
> 
> **Classification losses**
> - 0/1: $\ell = \mathbb{1}[\hat{y} \ne y]$
> - Log: $\ell = -\log p(y \mid x)$ — proper scoring rule.
> - Hinge: $\ell = \max(0, 1 - y \cdot f(x))$ with $y \in \{-1, +1\}$.
> - Exponential: $\ell = e^{-y \cdot f(x)}$ — used by AdaBoost.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements five loss functions in pure NumPy — MSE, MAE,
> Huber (regression) and log-loss, hinge (classification) — then evaluates each
> on a tiny toy batch.
> 
> **Why it matters:** Every model you train is minimising *one of these five*.
> Knowing their shapes and gradients by heart is the difference between
> choosing a loss deliberately and defaulting to MSE by accident.
> 
> **What to watch for:** `np.clip(p, eps, 1 - eps)` in `log_loss` is *essential* —
> `log(0)` is `-inf`, and a single over-confident prediction would otherwise
> make the whole loss `NaN`. Always clip probabilities before taking logs.


### Code cell #1
```python
import numpy as np

def mse(y, yhat):     return np.mean((y - yhat) ** 2)
def mae(y, yhat):     return np.mean(np.abs(y - yhat))
def huber(y, yhat, delta=1.0):
    e = np.abs(y - yhat)                          # absolute error
    quad = 0.5 * e ** 2                            # quadratic region: |e| <= delta
    lin  = delta * (e - 0.5 * delta)               # linear region: |e| > delta
    return np.mean(np.where(e <= delta, quad, lin))

def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)                   # avoid log(0) = -inf
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def hinge_loss(y, f):
    # y in {-1, +1}, f = raw score (decision function)
    return np.mean(np.maximum(0.0, 1.0 - y * f))

y = np.array([1.0, 2.0, 3.0])
yhat = np.array([1.5, 2.5, 2.0])
print("MSE :", mse(y, yhat))
print("MAE :", mae(y, yhat))
print("Huber:", huber(y, yhat, delta=1.0))

ybin = np.array([1, 0, 1])
p    = np.array([0.9, 0.1, 0.4])
print("log loss:", log_loss(ybin, p))

ysvm = np.array([1, -1, 1])
f    = np.array([2.0, -3.0, 0.5])
print("hinge:", hinge_loss(ysvm, f))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `mse`; def `mae`; def `huber`; def `log_loss`; def `hinge_loss`
- **Key imports:** `import numpy as np`

> **Takeaway:** MSE → Gaussian noise assumption; MAE → Laplace; Huber → the
> robust compromise; log-loss → the proper scoring rule for probabilities;
> hinge → the maximum-margin objective of SVMs. Pick the loss that matches
> *the noise model you believe in*, not the one that defaults first.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-computes the same losses with `sklearn.metrics` — the
> production-grade implementations you would actually ship.
> 
> **Why it matters:** The scratch version is for learning; the sklearn
> version is what you call in real pipelines. They should agree to many
> decimal places — if they do not, your scratch code has a bug.
> 
> **What to watch for:** `sklearn.metrics.log_loss` requires explicit
> `labels=[0, 1]` when `y` is not 0/1 floats, otherwise it tries to infer the
> label set from `p` and can produce surprising results on small inputs.


### Code cell #2
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    log_loss, hinge_loss

print("MSE :", mean_squared_error(y, yhat))
print("MAE :", mean_absolute_error(y, yhat))
print("log :", log_loss(ybin, p, labels=[0, 1]))   # explicit label set
print("hinge:", hinge_loss(ysvm, f))
```

**What this code does:**

- **Headline:** `from sklearn.metrics import mean_squared_error, mean_absolute_error, \`
- **Key imports:** `from sklearn.metrics import mean_squared_error`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a 200-row, 1-feature regression dataset with
> known true coefficient, then shifts the target by 50 so the units look like
> real-valued predictions (e.g. house prices).
> 
> **Why it matters:** A dataset with a *known* ground-truth coefficient lets
> us check whether OLS and Huber recover it — and whether outliers push the
> estimate away from the truth.
> 
> **What to watch for:** `coef=True` returns the true coefficient used to
> generate the data; we compare our fit coefficients against it as a sanity
> check.


### Code cell #3
```python
from sklearn.datasets import make_regression

X, y, coef = make_regression(n_samples=200, n_features=1, noise=15,
                              coef=True, random_state=0)   # return ground truth
y = y + 50  # shift to make units interpretable (e.g. dollars, scores)
print("target range:", y.min().round(1), "to", y.max().round(1))
print("true coef:", coef)
```

**What this code does:**

- **Headline:** `from sklearn.datasets import make_regression`
- **Key imports:** `from sklearn.datasets import make_regression`
- **Notable calls:** `make_regression`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots MSE, MAE, and Huber as functions of the prediction
> error `e = y - y_hat` over the range [-5, 5]. The Huber curve is quadratic
> near 0 and linear far away — the visual definition of robust loss.
> 
> **Why it matters:** Seeing the three curves on the same axes is the fastest
> way to internalise *why* MSE blows up under outliers (the parabola) and *why*
> MAE is non-differentiable at 0 (the V shape). Huber splices both.
> 
> **What to watch for:** the `rf"..."` raw f-string lets us put a LaTeX
> `$\delta$` into the legend without Python trying to interpret the backslash.


### Code cell #4
```python
import matplotlib.pyplot as plt

errs = np.linspace(-5, 5, 200)             # prediction errors to plot loss over
plt.figure(figsize=(7, 4))
plt.plot(errs, errs ** 2,       label="MSE", lw=2)
plt.plot(errs, np.abs(errs),    label="MAE", lw=2)
delta = 1.0
plt.plot(errs, np.where(np.abs(errs) <= delta, 0.5 * errs**2,
                        delta * (np.abs(errs) - 0.5 * delta)),
         label=rf"Huber ($\delta$={delta})", lw=2)
plt.axhline(0, color="gray", lw=0.5)        # zero line for reference
plt.xlabel(r"error $y - \hat{y}$"); plt.ylabel("loss")
plt.title("Regression loss functions"); plt.legend(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> **Takeaway:** MSE = parabola (smooth, but explodes for large errors); MAE
> = V-shape (robust, but kinked at 0); Huber = quadratic near 0, linear far
> away — the smooth, robust compromise that gradient methods like.

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> **What this does:** Splits the regression data into train/test, fits OLS
> (MSE minimiser) and HuberRegressor (Huber minimiser), then prints test MSE
> and MAE for each.
> 
> **Why it matters:** On *clean* data, OLS should win on MSE — it is the
> maximum-likelihood estimator under Gaussian noise. The Huber regressor is
> designed for *contaminated* data, so the comparison here sets up the
> outlier experiment in the next section.
> 
> **What to watch for:** `HuberRegressor(epsilon=1.35)` — the `epsilon`
> controls where quadratic transitions to linear; 1.35 is the value chosen
> in the original paper as a good default.


### Code cell #5
```python
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

ols = LinearRegression().fit(X_tr, y_tr)               # MSE minimiser
hub = HuberRegressor(epsilon=1.35).fit(X_tr, y_tr)     # Huber minimiser

for name, model in [("OLS", ols), ("Huber", hub)]:
    p = model.predict(X_te)
    print(f"{name}: MSE={mean_squared_error(y_te, p):.2f} "
          f"MAE={mean_absolute_error(y_te, p):.2f}")
```

**What this code does:**

- **Headline:** `from sklearn.linear_model import LinearRegression, HuberRegressor`
- **Key imports:** `from sklearn.linear_model import LinearRegression`, `from sklearn.model_selection import train_test_split`
- **Notable calls:** `LinearRegression`, `train_test_split`

> **Takeaway:** OLS is the MLE under Gaussian noise — unbeatable on clean
> data, but its coefficients are *dragged* by any outlier. Huber trades a
> small efficiency loss on clean data for substantial protection against
> contamination, which is the realistic case.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add 10% outlier points to the regression target — which loss function's
>    model coefficients drift the least?
> 2. Plot log-loss and hinge-loss on the same axes over $f \in [-3, 3]$ — why
>    is hinge loss said to focus on the margin?
> 3. Show that MSE loss corresponds to a Gaussian noise assumption and that
>    MAE corresponds to a Laplace noise assumption.

> **What this does:** Injects 10% outlier points (y += 200) into the target,
> refits OLS and Huber on the contaminated data, and prints the recovered
> slopes against the known truth.
> 
> **Why it matters:** This is the textbook demonstration of *why* robust
> losses exist. OLS's slope gets dragged by the outliers; Huber's slope
> barely moves. A 10-second experiment that justifies an entire family of
> loss functions.
> 
> **What to watch for:** `rng.choice(len(y), size=20, replace=False)` picks
> 20 distinct outlier indices — without `replace=False` we might pick the
> same row twice and double-count its contamination.


### Code cell #6
```python
# Inject 10% outliers and compare OLS vs Huber.
rng = np.random.default_rng(0)
y_outlier = y.copy()                            # never mutate the original
out_idx = rng.choice(len(y), size=20, replace=False)   # 20 distinct rows
y_outlier[out_idx] += 200                       # add a large positive offset

ols = LinearRegression().fit(X, y_outlier)
hub = HuberRegressor(epsilon=1.35).fit(X, y_outlier)
print(f"OLS slope={ols.coef_[0]:.3f}  Huber slope={hub.coef_[0]:.3f}  (true={coef:.3f})")
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(0)`
- **Notable calls:** `LinearRegression`

> **Takeaway:** Robust losses (Huber, MAE, quantile) trade a small amount of
> efficiency on clean Gaussian data for *a lot* of protection against outliers.
> Use them whenever you cannot guarantee your labels are clean.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement the *quantile loss* and use it for a 90th-percentile regression.
> 2. Implement *focal loss* and explain when it helps over cross-entropy.
> 3. Show empirically that minimising MSE is equivalent to maximum-likelihood
>    under a Gaussian noise model.
> 4. Implement the *Kullback–Leibler divergence* as a loss between two discrete distributions.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why is MSE more sensitive to outliers than MAE?
> 2. What is a *proper scoring rule*, and why is log-loss one?
> 3. Why does SVM use hinge loss instead of 0/1 loss?
> 4. What is the gradient of MSE and MAE — and why does the MAE gradient cause issues for SGD?
> 5. When would you use Huber loss over MSE?
> 6. Why is cross-entropy the natural loss for softmax classifiers?
> 7. What is the relationship between negative log-likelihood and KL divergence?
> 8. Why is the 0/1 loss not directly minimisable by gradient methods?
