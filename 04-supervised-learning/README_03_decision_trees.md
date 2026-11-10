# Code Walkthrough — `03_decision_trees.ipynb`

> Module: **04-supervised-learning** | Notebook: **03_decision_trees.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 24
- **Markdown cells:** 19
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

> # 03 — Decision Trees
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
> ## 1. Concept Overview — Decision trees — human-readable, non-parametric

> A decision tree partitions feature space into axis-aligned rectangles, with
> a prediction at each leaf. Trees are the building block of many modern
> ensemble methods (random forests, gradient boosting) and are uniquely
> interpretable.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> At each node, we choose the feature and threshold that maximally reduce
> impurity:
> 
> - **Gini**: $1 - \sum_k p_k^2$ (probability that two random samples have different labels).
> - **Entropy**: $-\sum_k p_k \log p_k$ (information-theoretic).
> - **MSE** (regression): node variance.
> 
> Gain = $I(\text{parent}) - \sum_j \frac{N_j}{N} I(\text{child}_j)$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

def best_split(X, y, feature):
    order = np.argsort(X[:, feature])
    xs, ys = X[order, feature], y[order]
    best_gain, best_thr = 0, None
    parent_imp = gini(y)
    for i in range(1, len(y)):
        if xs[i] == xs[i-1]:
            continue
        left, right = ys[:i], ys[i:]
        g = parent_imp - (len(left)/len(y))*gini(left) - (len(right)/len(y))*gini(right)
        if g > best_gain:
            best_gain, best_thr = g, (xs[i-1] + xs[i]) / 2
    return best_gain, best_thr

class Node:
    def __init__(self):
        self.is_leaf = False
    def predict_one(self, x):
        return (self.prediction if self.is_leaf
                else self.left.predict_one(x) if x[self.feat] <= self.thr
                else self.right.predict_one(x))

class TreeScratch:
    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples
    def fit(self, X, y, depth=0):
        self.node = Node()
        if depth >= self.max_depth or len(y) < self.min_samples or gini(y) == 0:
            self.node.is_leaf = True
            vals, cnts = np.unique(y, return_counts=True)
            self.node.prediction = vals[np.argmax(cnts)]
            return self
        # find best (feature, threshold)
        best = max(
            (best_split(X, y, f) + (f,) for f in range(X.shape[1])),
            key=lambda t: t[0]
        )
        gain, thr, feat = best
        if gain == 0:
            self.node.is_leaf = True
            vals, cnts = np.unique(y, return_counts=True)
            self.node.prediction = vals[np.argmax(cnts)]
            return self
        self.node.feat, self.node.thr = feat, thr
        m = X[:, feat] <= thr
        self.node.left  = TreeScratch(self.max_depth, self.min_samples).fit(X[m],  y[m],  depth+1)
        self.node.right = TreeScratch(self.max_depth, self.min_samples).fit(X[~m], y[~m], depth+1)
        return self
    def predict_one(self, x):
        return self.node.predict_one(x)
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 2))
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
tree = TreeScratch(max_depth=4).fit(X, y)
print("train accuracy:", (tree.predict(X) == y).mean())
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `gini`; def `best_split`; class `Node`; class `TreeScratch`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for name, tree in [
    ("depth=1", DecisionTreeClassifier(max_depth=1, random_state=0)),
    ("depth=3", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ("depth=None", DecisionTreeClassifier(random_state=0)),  # full-grown
]:
    tree.fit(X_tr, y_tr)
    print(f"{name:11s} train={accuracy_score(y_tr, tree.predict(X_tr)):.3f}  "
          f"test={accuracy_score(y_te, tree.predict(X_te)):.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.tree import DecisionTreeClassifier, plot_tree`
- **Key imports:** `from sklearn.tree import DecisionTreeClassifier`, `from sklearn.datasets import load_iris`, `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import accuracy_score`
- **Notable calls:** `DecisionTree`, `train_test_split`, `load_iris`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> We use the titanic dataset to demonstrate how a tree handles missing
> categorical data without preprocessing.


### Code cell #3
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[cols].copy()
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna("S")
for c in ["Sex", "Embarked"]:
    X[c] = LabelEncoder().fit_transform(X[c])
y = titanic["Survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
print(f"train={tree.score(X_tr, y_tr):.3f}  test={tree.score(X_te, y_te):.3f}")
print("feature importances:", dict(zip(X.columns, tree.feature_importances_.round(3))))
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`, `from sklearn.preprocessing import LabelEncoder`
- **Notable calls:** `DecisionTree`, `train_test_split`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plot_tree(tree, feature_names=X.columns, class_names=["died", "survived"],
          filled=True, rounded=True, max_depth=3)
plt.title("First 3 levels of the titanic decision tree")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Trees have an extreme bias/variance trade-off. Depth controls it:
> 
> - Low depth → high bias, low variance (under-fitting).
> - High depth → low bias, high variance (over-fitting).
> - The right depth is found by cross-validation.


### Code cell #5
```python
from sklearn.model_selection import cross_val_score
import numpy as np
depths = range(1, 16)
tr_scores, cv_scores = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=0)
    t.fit(X_tr, y_tr)
    tr_scores.append(t.score(X_tr, y_tr))
    cv_scores.append(cross_val_score(t, X, y, cv=5).mean())

plt.plot(depths, tr_scores, label="train")
plt.plot(depths, cv_scores, label="5-fold CV")
plt.xlabel("max depth"); plt.ylabel("accuracy"); plt.legend()
plt.title("Bias-variance trade-off in decision-tree depth")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.model_selection import cross_val_score`
- **Key imports:** `from sklearn.model_selection import cross_val_score`, `import numpy as np`
- **Notable calls:** `DecisionTree`, `cross_val_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace Gini with entropy — does the chosen tree differ much?
> 2. Try `min_samples_leaf=10` — how does the train-test gap change?
> 3. Use `ccp_alpha` (cost-complexity pruning) and find the value that maximises CV accuracy.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a **regression tree** from scratch using MSE as the splitting criterion.
> 2. Implement **pre-pruning** via `min_samples_split` and `max_leaf_nodes`.
> 3. Implement **post-pruning** via cost-complexity pruning.
> 4. Implement a tree that handles missing values via surrogate splits.
> 5. Build an `explain_path` function that returns the rule path for a single prediction.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. How does a decision tree decide where to split?
> 2. What is the difference between Gini and entropy? When does it matter?
> 3. Why are decision trees prone to over-fitting, and how do you prevent it?
> 4. Why are trees invariant to feature scaling?
> 5. What is feature importance in a tree, and what are its caveats?
> 6. How do you handle categorical variables in a tree?
> 7. What is cost-complexity pruning?
> 8. Why are axis-aligned splits a limitation, and how do oblique trees address it?
