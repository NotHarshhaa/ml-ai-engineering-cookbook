# Code Walkthrough — `04_random_forest.ipynb`

> Module: **04-supervised-learning** | Notebook: **04_random_forest.ipynb**
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

> # 04 — Random Forests
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
> ## 1. Concept Overview — Random forests — bagging + random subspaces

> A random forest is an ensemble of decision trees. Each tree is trained on a
> bootstrap sample of the data, and at each split only a random subset of
> features is considered. The two sources of randomness decorrelate the trees,
> making the average much more stable than any single tree.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Bagging** variance reduction: if $\hat{f}_1, \ldots, \hat{f}_B$ are
> i.i.d. with variance $\sigma^2$, then $\bar{f} = \frac{1}{B}\sum_b \hat{f}_b$
> has variance $\sigma^2 / B$. Real trees are correlated, so the reduction is
> $\rho \sigma^2 + (1-\rho) \sigma^2/B$ — feature sub-sampling lowers $\rho$.
> 
> **Out-of-bag (OOB) error** — each tree leaves ~37% of samples untouched.
> Use those for unbiased validation.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestScratch:
    def __init__(self, n_estimators=20, max_features="sqrt",
                 max_depth=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)
    def fit(self, X, y):
        self.trees = []
        n, d = X.shape
        k = int(np.sqrt(d)) if self.max_features == "sqrt" else d
        self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n, n)   # bootstrap sample
            feats = self.rng.choice(d, k, replace=False)
            t = DecisionTreeClassifier(max_depth=self.max_depth,
                                        random_state=int(self.rng.integers(1e6)))
            t.fit(X[idx][:, feats], y[idx])
            self.trees.append((t, feats))
        return self
    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        for t, feats in self.trees:
            p = t.predict_proba(X[:, feats])
            # reindex columns to match classes_
            for j, c in enumerate(t.classes_):
                out[:, list(self.classes_).index(c)] += p[:, j]
        return out / len(self.trees)
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                            random_state=0)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
rf = RandomForestScratch(n_estimators=50, max_features="sqrt",
                          max_depth=8, random_state=0).fit(X_tr, y_tr)
print("scratch RF accuracy:", (rf.predict(X_te) == y_te).mean())
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** class `RandomForestScratch`
- **Key imports:** `import numpy as np`, `from sklearn.tree import DecisionTreeClassifier`, `from sklearn.datasets import make_classification`, `from sklearn.model_selection import train_test_split`
- **Notable calls:** `DecisionTree`, `RandomForest`, `train_test_split`, `make_classification`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=200, max_depth=8, max_features="sqrt",
                            oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
print(f"train acc = {accuracy_score(y_tr, rf.predict(X_tr)):.3f}")
print(f"test  acc = {accuracy_score(y_te, rf.predict(X_te)):.3f}")
print(f"OOB   acc = {rf.oob_score_:.3f}")
print("feature importances:", dict(zip([f"x{i}" for i in range(X.shape[1])],
                                       rf.feature_importances_.round(3))))
```

**What this code does:**

- **Headline:** `from sklearn.ensemble import RandomForestClassifier`
- **Key imports:** `from sklearn.ensemble import RandomForestClassifier`, `from sklearn.metrics import accuracy_score`
- **Notable calls:** `RandomForest`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1,
                            random_state=0).fit(X_tr, y_tr)
print(f"R^2 = {rf.score(X_te, y_te):.3f}")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
print("feature importances (ascending):")
print(imp.round(4))
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`, `from sklearn.datasets import fetch_california_housing`, `from sklearn.ensemble import RandomForestRegressor`
- **Notable calls:** `RandomForest`, `train_test_split`, `fetch_california_housing`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
imp.plot(kind="barh")
plt.title("Random-forest feature importances (California housing)")
plt.tight_layout(); plt.show()

# Number of trees vs OOB error
from sklearn.datasets import make_classification
Xc, yc = make_classification(n_samples=500, n_features=20, n_informative=8, random_state=0)
oobs = []
n_trees = [1, 2, 5, 10, 25, 50, 100, 200, 400]
for n in n_trees:
    r = RandomForestClassifier(n_estimators=n, oob_score=True,
                                random_state=0, n_jobs=-1).fit(Xc, yc)
    oobs.append(1 - r.oob_score_)
plt.figure(figsize=(7, 4))
plt.plot(n_trees, oobs, "o-")
plt.xlabel("# trees"); plt.ylabel("OOB error")
plt.title("OOB error decreases with more trees")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `from sklearn.datasets import make_classification`
- **Notable calls:** `RandomForest`, `make_classification`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For a forest we evaluate:
> 
> - **OOB error** — internal, no held-out set needed.
> - **Permutation importance** — drop a feature, measure accuracy loss.
> - **Bias/variance** — trees *do* over-fit individually; the ensemble barely.


### Code cell #5
```python
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0,
                              n_jobs=-1)
pm = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
pm.plot(kind="barh")
plt.title("Permutation importance (California housing)")
plt.xlabel("drop in R^2")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.inspection import permutation_importance`
- **Key imports:** `from sklearn.inspection import permutation_importance`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Set `max_features=None` (all features at each split) — does the forest
>    still benefit from many trees? Why does it behave like bagging?
> 2. Reduce `min_samples_leaf` from 1 to 50 — what happens to the OOB curve?
> 3. Compare `RandomForestRegressor` with `ExtraTreesRegressor` on the same task.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **bagging** (bootstrap aggregating) of logistic regressions.
> 2. Implement **out-of-bag error** from scratch and verify against sklearn.
> 3. Implement **proximity matrix** from a random forest and use it for clustering.
> 4. Implement **partial dependence plots** from scratch for the top feature.
> 5. Build a *warm-start* loop that adds 10 trees at a time and plots OOB error.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. How does a random forest differ from bagging?
> 2. Why does feature sub-sampling reduce correlation between trees?
> 3. What is OOB error, and when is it preferable to cross-validation?
> 4. How are feature importances computed in a random forest?
> 5. Why is permutation importance more reliable than impurity importance?
> 6. When would a random forest underperform gradient boosting?
> 7. How do you handle class imbalance in a random forest?
> 8. Why does a random forest not extrapolate beyond the training range?
