# Code Walkthrough — `02_feature_selection.ipynb`

> Module: **03-feature-engineering** | Notebook: **02_feature_selection.ipynb**
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

> # 02 — Feature Selection
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
> ## 1. Concept Overview — Feature selection — keeping what helps, dropping what hurts

> More features are not always better. Irrelevant or redundant features add
> noise, slow training, and harm generalisation. **Feature selection**
> identifies the subset that maximises downstream performance.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Three families of methods:
> 
> - **Filter** — score each feature independently of any model (correlation,
>   mutual information, $\chi^2$).
> - **Wrapper** — search over subsets, training a model for each
>   (forward / backward selection, RFE).
> - **Embedded** — selection happens inside the training (L1 / Lasso,
>   tree-based feature importances).
> 
> Mutual information: $I(X; Y) = \sum_{x,y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 300
signal = rng.standard_normal(n)
y = 2 * signal + 0.5 * rng.standard_normal(n)        # depends on signal only

X = pd.DataFrame({
    "useful":   signal,
    "noise1":   rng.standard_normal(n),
    "noise2":   rng.standard_normal(n),
    "redundant": signal + 0.01 * rng.standard_normal(n),  # collinear with useful
    "constant": np.ones(n),
})
X.head()
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `import pandas as pd`


### Code cell #2
```python
# Pearson correlation filter from scratch.
def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return np.sum(x*y) / (np.sqrt(np.sum(x*x)) * np.sqrt(np.sum(y*y)))

correlations = {col: pearson(X[col].values, y) for col in X.columns}
print("Pearson correlation with target:")
for k, v in sorted(correlations.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k:10s}  {v:+.3f}")

# Variance filter — drop near-constant columns.
variances = X.var()
print("\nvariances:"); print(variances.round(4))
```

**What this code does:**

- **Headline:** `def pearson(x, y):`
- **Defines:** def `pearson`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #3
```python
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel,
)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

# Filter (top-2 by F-statistic)
filter_top2 = SelectKBest(f_regression, k=2).fit(X, y)
print("Filter top-2:", X.columns[filter_top2.get_support()].tolist())

# Filter (top-2 by mutual information)
mi_top2 = SelectKBest(mutual_info_regression, k=2).fit(X, y)
print("MI top-2:   ", X.columns[mi_top2.get_support()].tolist())

# Wrapper (recursive feature elimination, keep 2)
rfe = RFE(LinearRegression(), n_features_to_select=2).fit(X, y)
print("RFE top-2:  ", X.columns[rfe.get_support()].tolist())

# Embedded (Lasso L1)
lasso = Lasso(alpha=0.05).fit(X, y)
print("Lasso coefs:", dict(zip(X.columns, lasso.coef_.round(3))))

# Embedded (random forest importance)
rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRF importances:\n", imp.round(3))
```

**What this code does:**

- **Headline:** `from sklearn.feature_selection import (`
- **Key imports:** `from sklearn.feature_selection import (`, `from sklearn.linear_model import LinearRegression`, `from sklearn.ensemble import RandomForestRegressor`
- **Notable calls:** `LinearRegression`, `RandomForest`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #4
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
print("shape:", X.shape)
print(X.head(3))

# Add 5 noisy features so the selector has work to do.
rng = np.random.default_rng(0)
for i in range(5):
    X[f"noise_{i}"] = rng.standard_normal(len(X))
print("extended shape:", X.shape)
```

**What this code does:**

- **Headline:** `from sklearn.datasets import fetch_california_housing`
- **Key imports:** `from sklearn.datasets import fetch_california_housing`
- **Notable calls:** `fetch_california_housing`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #5
```python
import matplotlib.pyplot as plt
import seaborn as sns

corr = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
plt.title("Feature correlation heatmap — look for collinear pairs")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> The honest way to evaluate feature selection is **nested cross-validation**:
> the inner loop selects features, the outer loop scores the resulting model.
> Doing selection on the full dataset and then CV scoring the same model is a
> classic leakage mistake.


### Code cell #6
```python
from sklearn.model_selection import cross_val_score, Pipeline

# Naive (leaky): select on full data, then CV
selector = SelectKBest(mutual_info_regression, k=5)
X_sel = selector.fit_transform(X, y)
naive = cross_val_score(LinearRegression(), X_sel, y, cv=5).mean()

# Proper: selection inside the CV pipeline.
pipe = Pipeline([
    ("select", SelectKBest(mutual_info_regression, k=5)),
    ("model",  LinearRegression()),
])
proper = cross_val_score(pipe, X, y, cv=5).mean()

print(f"naive (leaky): R^2 = {naive:.3f}")
print(f"proper:       R^2 = {proper:.3f}")
print("(Proper is what you would actually deploy with.)")
```

**What this code does:**

- **Headline:** `from sklearn.model_selection import cross_val_score, Pipeline`
- **Key imports:** `from sklearn.model_selection import cross_val_score`
- **Notable calls:** `LinearRegression`, `cross_val_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add 30 noise features and rerun Lasso — at what $\alpha$ does it start
>    to discard signal features too?
> 2. Compare RFE with a LogisticRegression vs with a RandomForest — which is
>    faster and which is more accurate?
> 3. Replace `SelectKBest` with `SelectPercentile(50%)` — does it matter on this dataset?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement forward-stepwise selection from scratch on a small dataset.
> 2. Implement a permutation-importance selector (drop feature, re-score, repeat).
> 3. Build a sklearn `Pipeline` that does selection + model and use it in `GridSearchCV`.
> 4. Implement Boruta — compare all-features vs only-shadow-important features.
> 5. Write a `feature_selection_report(X, y)` that returns one tidy DataFrame
>    with F-stat, MI, L1-coef, RF-importance.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between filter, wrapper, and embedded methods?
> 2. Why is mutual information more general than Pearson correlation?
> 3. What is feature leakage through selection and how do you avoid it?
> 4. How does L1 regularisation induce sparsity?
> 5. What is the *curse of dimensionality*, and how does feature selection help?
> 6. When is RFE preferable to `SelectKBest`?
> 7. Why is permutation importance considered more reliable than RF importance?
> 8. How would you select features for a 1M-dimensional sparse text feature matrix?
