# Code Walkthrough — `03_feature_scaling.ipynb`

> Module: **03-feature-engineering** | Notebook: **03_feature_scaling.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 23
- **Markdown cells:** 18
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

> # 03 — Feature Scaling
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
> ## 1. Concept Overview — Feature scaling — putting features on the same footing

> Many models — kNN, SVM, gradient descent, regularised regression — compare
> or combine features via distance or dot product. If one feature is measured
> in thousands and another in fractions, the model will silently over-weight the
> former. **Scaling** fixes this.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Four common scalers:
> 
> - **Min-Max**: $\tilde{x} = (x - x_{\min}) / (x_{\max} - x_{\min}) \in [0, 1]$
> - **Standard (z-score)**: $\tilde{x} = (x - \mu) / \sigma$
> - **Robust**: $\tilde{x} = (x - \text{median}) / \text{IQR}$ — robust to outliers.
> - **Unit-norm**: scale each *sample* so $\|x_i\|_2 = 1$.
> 
> Which one you use depends on the model and the data distribution.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 1, 200),
                rng.normal(50, 10, 200),
                rng.normal(1000, 200, 200)]).T
print("before scaling, column means/stds:")
print("means:", X.mean(axis=0).round(2))
print("stds :", X.std(axis=0).round(2))

def standardise(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def minmax(X):
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / (hi - lo)

def robust_scale(X):
    med = np.median(X, axis=0)
    q1  = np.quantile(X, 0.25, axis=0)
    q3  = np.quantile(X, 0.75, axis=0)
    return (X - med) / (q3 - q1)

print("\nafter standardise:"); print("means:", standardise(X).mean(0).round(3))
print("\nafter minmax:");      print("min/max:", minmax(X).min(0).round(3), minmax(X).max(0).round(3))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `standardise`; def `minmax`; def `robust_scale`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
)
import pandas as pd

scalers = {
    "Standard": StandardScaler(),
    "MinMax":   MinMaxScaler(),
    "Robust":   RobustScaler(),
    "MaxAbs":   MaxAbsScaler(),
}
for name, s in scalers.items():
    Xt = s.fit_transform(X)
    print(f"{name:10s} mean={Xt.mean(0).round(3)} std={Xt.std(0).round(3)}")
```

**What this code does:**

- **Headline:** `from sklearn.preprocessing import (`
- **Key imports:** `from sklearn.preprocessing import (`, `import pandas as pd`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
cols = ["class","alcohol","malic_acid","ash","alcalinity","mg","phenols","flavanoids",
        "nonflav_phenols","proanth","color_int","hue","od280","proline"]
wine = pd.read_csv(wine_url, names=cols)
X = wine.drop(columns="class")
y = wine["class"]
print("shape:", X.shape)
print("original column ranges (min/max):")
print((X.describe().loc[["min", "max"]].T.assign(range=lambda d: d["max"]-d["min"])))
```

**What this code does:**

- **Headline:** `wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=X, ax=axes[0]); axes[0].set_title("Before scaling"); axes[0].set_xticklabels([])
sns.boxplot(data=StandardScaler().fit_transform(X), ax=axes[1])
axes[1].set_title("After Standard scaling"); axes[1].set_xticklabels([])
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> kNN is the textbook *scaling-sensitive* model. We compare accuracy on the
> wine dataset across the four scalers.


### Code cell #5
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

for name, scaler in scalers.items():
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scale", scaler), ("knn", KNeighborsClassifier(5))])
    acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
    print(f"{name:10s} kNN accuracy = {acc:.3f}")

# No scaling at all:
acc_no = cross_val_score(KNeighborsClassifier(5), X, y, cv=5, scoring="accuracy").mean()
print(f"no-scaling  kNN accuracy = {acc_no:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.neighbors import KNeighborsClassifier`
- **Key imports:** `from sklearn.neighbors import KNeighborsClassifier`, `from sklearn.model_selection import cross_val_score`, `from sklearn.pipeline import Pipeline`
- **Notable calls:** `cross_val_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace kNN with a `DecisionTreeClassifier` — do the scalers still matter?
>    Why or why not?
> 2. Inject 5% outliers into `proline` — which scaler keeps the inlier pattern intact?
> 3. Train an SVM with RBF kernel — how sensitive is it to scaling?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement `QuantileTransformer` from scratch (map each feature to a uniform).
> 2. Build a `Pipeline` that scales only numerical columns, leaving binary ones untouched.
> 3. Write a function `pick_scaler(X, model)` that heuristically recommends a scaler.
> 4. Implement `PowerTransformer` (Yeo-Johnson) from scratch using `scipy.optimize`.
> 5. Demonstrate that scaling improves SGD convergence rate on a logistic regression.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Which models are sensitive to feature scaling, and which are not? Why?
> 2. What is the difference between `StandardScaler` and `RobustScaler`?
> 3. When is `MinMaxScaler` a poor choice?
> 4. Why is scaling fit on the training set only?
> 5. Does scaling affect tree-based models? Justify.
> 6. What is `Normalizer` and how does it differ from the other scalers?
> 7. How does scaling affect gradient-descent convergence?
> 8. How would you scale features in an online-learning setting?
