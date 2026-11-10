# Code Walkthrough — `01_linear_regression.ipynb`

> Module: **04-supervised-learning** | Notebook: **01_linear_regression.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 24
- **Markdown cells:** 18
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

> # 01 — Linear Regression
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
> ## 1. Concept Overview — Linear regression — the workhorse

> Linear regression fits $y = X\beta + \epsilon$ by minimising the residual
> sum of squares. It is the first model everyone learns because every more
> complex model is best understood by how it *departs* from this baseline.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Ordinary Least Squares (OLS)**
> 
> $$\hat{\beta} = (X^\top X)^{-1} X^\top y$$
> 
> Loss: $\; L(\beta) = \|X\beta - y\|^2$
> Gradient: $\;\nabla_\beta L = 2 X^\top (X\beta - y)$
> 
> **Ridge (L2)**: $\hat{\beta} = (X^\top X + \lambda I)^{-1} X^\top y$
> 
> **Lasso (L1)**: $\arg\min_\beta \|X\beta - y\|^2 + \lambda \|\beta\|_1$ — no closed form, subgradient.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

class LinearRegressionScratch:
    def fit(self, X, y):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        self.beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return self
    def predict(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return Xb @ self.beta

class RidgeScratch:
    def __init__(self, lam=1.0):
        self.lam = lam
    def fit(self, X, y):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        k = Xb.shape[1]
        reg = self.lam * np.eye(k)
        reg[0, 0] = 0  # do not regularise intercept
        A = Xb.T @ Xb + reg
        self.beta = np.linalg.solve(A, Xb.T @ y)
        return self
    def predict(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return Xb @ self.beta

def gd_linear(X, y, lr=0.05, steps=300):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.zeros(Xb.shape[1])
    for _ in range(steps):
        g = Xb.T @ (Xb @ beta - y) * 2 / len(y)
        beta = beta - lr * g
    return beta

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 2))
y = 3 * X[:, 0] - 1.5 * X[:, 1] + 2 + 0.1 * rng.standard_normal(100)

ols = LinearRegressionScratch().fit(X, y)
print("OLS beta:", ols.beta.round(3))
print("Ridge(0.1) beta:", RidgeScratch(0.1).fit(X, y).beta.round(3))
print("GD beta:      ", gd_linear(X, y, lr=0.1, steps=500).round(3))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** class `LinearRegressionScratch`; class `RidgeScratch`; def `gd_linear`
- **Key imports:** `import numpy as np`
- **Notable calls:** `LinearRegression`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=20, n_informative=5,
                       noise=10.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

models = {
    "OLS":    LinearRegression(),
    "Ridge":  Ridge(alpha=1.0),
    "Lasso":  Lasso(alpha=0.1),
    "Elastic":ElasticNet(alpha=0.1, l1_ratio=0.5),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    print(f"{name:7s} R^2 = {r2_score(y_te, p):.3f}  RMSE = {np.sqrt(mean_squared_error(y_te, p)):.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet`
- **Key imports:** `from sklearn.linear_model import LinearRegression`, `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import r2_score`, `from sklearn.datasets import make_regression`
- **Notable calls:** `LinearRegression`, `train_test_split`, `make_regression`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
import pandas as pd
housing = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)
print("shape:", housing.shape)
print(housing.head(3))
print("\nfeature-target correlation with medv:")
print(housing.corr()["medv"].abs().sort_values(ascending=False).head(8))
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
import matplotlib.pyplot as plt
import seaborn as sns

X = housing.drop(columns="medv"); y = housing["medv"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
ols = LinearRegression().fit(X_tr, y_tr)
preds = ols.predict(X_te)
resid = y_te - preds

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(preds, y_te, alpha=0.5)
axes[0].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("actual")
axes[0].set_title("Predicted vs actual")

axes[1].scatter(preds, resid, alpha=0.5)
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual")
axes[1].set_title("Residual plot")

coefs = pd.Series(ols.coef_, index=X.columns).sort_values()
axes[2].barh(coefs.index, coefs.values)
axes[2].set_title("OLS coefficients")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`
- **Notable calls:** `LinearRegression`, `train_test_split`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Linear-regression metrics:
> 
> - **$R^2$**: $1 - \frac{SS_\text{res}}{SS_\text{tot}}$ — variance explained.
> - **RMSE / MAE**: error magnitude in target units.
> - **Adjusted $R^2$**: penalises adding irrelevant features.
> - **Residual analysis**: normality, homoscedasticity, independence.


### Code cell #5
```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([("sc", StandardScaler()),
                 ("rg", Ridge(alpha=1.0))])
scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
print(f"Ridge 5-fold R^2: mean={scores.mean():.3f} std={scores.std():.3f}")

# Adjusted R^2
n, p = X.shape
r2 = r2_score(y_te, ols.predict(X_te))
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R^2 = {r2:.3f}  adjusted R^2 = {r2_adj:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.model_selection import cross_val_score`
- **Key imports:** `from sklearn.model_selection import cross_val_score`, `from sklearn.preprocessing import StandardScaler`, `from sklearn.pipeline import Pipeline`
- **Notable calls:** `cross_val_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a quadratic feature `lstat^2` and re-fit — does it improve the residual plot?
> 2. Sweep `alpha` for Lasso from $10^{-4}$ to $10^{1}$ — plot the coefficient path.
> 3. Standardise features and re-fit Ridge — does the coefficient picture change?


### Code cell #6
```python
import numpy as np
alphas = np.logspace(-4, 1, 30)
coefs_path = []
for a in alphas:
    Lasso(alpha=a, max_iter=5000).fit(X_tr, y_tr)
    coefs_path.append(Lasso(alpha=a, max_iter=5000).fit(X_tr, y_tr).coef_)
coefs_path = np.array(coefs_path)

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(coefs_path.shape[1]):
    ax.plot(alphas, coefs_path[:, i], label=X.columns[i])
ax.set_xscale("log")
ax.set_xlabel("alpha (regularization strength)")
ax.set_ylabel("coefficient")
ax.set_title("Lasso coefficient path")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **Huber regression** from scratch and compare against OLS on
>    outlier-injected data.
> 2. Implement **weighted least squares** where each point has a weight $w_i$.
> 3. Implement **quantile regression** at $q=0.9$ using an LP solver (`scipy.optimize.linprog`).
> 4. Implement **Bayesian linear regression** with a Gaussian prior (equivalent to Ridge)
>    and return posterior mean and variance.
> 5. Implement **recursive least squares** for online updates.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What are the Gauss-Markov assumptions, and what does BLUE mean?
> 2. Why does multicollinearity hurt OLS, and how does Ridge help?
> 3. What is the difference between Ridge and Lasso?
> 4. Why is the intercept usually not regularised?
> 5. What does an $R^2$ of 0.7 mean, and what are its failure modes?
> 6. Why look at a residual plot instead of just $R^2$?
> 7. When would you prefer MAE over RMSE?
> 8. How does standardising features change the interpretation of Ridge coefficients?
