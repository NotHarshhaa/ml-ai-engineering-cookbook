# Code Walkthrough — `04_feature_encoding.ipynb`

> Module: **03-feature-engineering** | Notebook: **04_feature_encoding.ipynb**
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

> # 04 — Feature Encoding
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
> ## 1. Concept Overview — Feature encoding — turning categories into numbers

> Most real datasets include categorical variables: city, product type,
> browser, day-of-week. Most models consume numbers. **Encoding** is the
> mapping $\phi: c \mapsto \mathbb{R}^k$ that makes a category usable as a feature.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Encoders seen in practice:
> 
> - **One-hot**: $\phi(c) = e_c \in \{0,1\}^{|C|}$ — expressive but sparse.
> - **Ordinal**: $\phi(c) = \text{rank}(c)$ — only when the categories are ordered.
> - **Label / integer**: $\phi(c) = i$ — implies a meaningless ordering.
> - **Target (mean) encoding**: $\phi(c) = \bar{y}\,|_{x=c}$ — high-capacity, leakage-prone.
> - **Frequency / count encoding**: $\phi(c) = \#\{i : x_i = c\}$.
> - **Embedding**: $\phi(c) = E_c \in \mathbb{R}^d$ learned by the model.
> 
> Mean encoding risk: $\mathbb{E}[\bar{y}|x=c]$ leaks target information if not cross-validated.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
colors = rng.choice(["red", "green", "blue", "yellow"], 20)
y = np.array([1.0, 2.5, 3.0, 1.5] * 5) + 0.1 * rng.standard_normal(20)

def one_hot(cats):
    levels = sorted(set(cats))
    idx = {c: i for i, c in enumerate(levels)}
    M = np.zeros((len(cats), len(levels)))
    for i, c in enumerate(cats): M[i, idx[c]] = 1
    return M, levels

def target_encode(cats, y, smoothing=1.0):
    global_mean = y.mean()
    out = np.empty(len(cats))
    for i, c in enumerate(cats):
        mask = np.array(cats) == c
        nc = mask.sum()
        # smoothed mean to regularise small groups
        out[i] = (y[mask].sum() + smoothing * global_mean) / (nc + smoothing)
    return out

M, levels = one_hot(colors)
print("one-hot shape:", M.shape, " levels:", levels)
print("target encoded (first 8):", target_encode(colors, y, smoothing=2.0)[:8].round(2))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `one_hot`; def `target_encode`
- **Key imports:** `import numpy as np`, `import pandas as pd`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder  # pip install category_encoders

df = pd.DataFrame({"color": colors, "y": y})
ohe = OneHotEncoder(sparse_output=False).fit_transform(df[["color"]])
print("OneHotEncoder output shape:", ohe.shape)
print("first 3 rows:\n", ohe[:3])

te = TargetEncoder(smoothing=2.0).fit(df[["color"]], df["y"])
print("TargetEncoder mapping:\n", te.mapping)
```

**What this code does:**

- **Headline:** `from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder`
- **Key imports:** `from sklearn.preprocessing import OneHotEncoder`, `from category_encoders import TargetEncoder  # pip install category_encoders`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
ames_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
diamonds = pd.read_csv(ames_url)
print("shape:", diamonds.shape)
categorical = diamonds.select_dtypes("object").columns.tolist()
print("categorical columns:", categorical)
for c in categorical:
    print(f"  {c}: {diamonds[c].nunique()} levels -> {diamonds[c].unique()[:5]}")
```

**What this code does:**

- **Headline:** `ames_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=diamonds, x="cut", y="price", order=["Fair","Good","Very Good","Premium","Ideal"],
            ax=axes[0])
axes[0].set_title("Price by cut (ordinal!)")

sns.boxplot(data=diamonds, x="color", y="price",
            order=sorted(diamonds["color"].unique()), ax=axes[1])
axes[1].set_title("Price by color")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> We compare three encodings on a price regression with `RandomForestRegressor`,
> using the categorical columns `cut`, `color`, `clarity`.


### Code cell #5
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]
X = diamonds[cat_cols + num_cols]; y = diamonds["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

def make_pipeline(encoder):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", encoder, cat_cols),
            ("num", "passthrough", num_cols),
        ])),
        ("rf", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)),
    ])

results = {}
for name, enc in [("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                   ("Ordinal", OrdinalEncoder()),
                   ("Target", TargetEncoder(smoothing=2.0))]:
    pipe = make_pipeline(enc).fit(X_tr, y_tr)
    results[name] = r2_score(y_te, pipe.predict(X_te))

for k, v in results.items():
    print(f"{k:8s} R^2 = {v:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.ensemble import RandomForestRegressor`
- **Defines:** def `make_pipeline`
- **Key imports:** `from sklearn.ensemble import RandomForestRegressor`, `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import r2_score`, `from sklearn.pipeline import Pipeline`
- **Notable calls:** `RandomForest`, `train_test_split`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add `k`-fold target encoding that avoids leakage — does R² improve?
> 2. Replace `RandomForestRegressor` with `LinearRegression` — which encoder wins now?
> 3. Drop rare categories (frequency < 1%) before one-hot encoding — what changes?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement leave-one-out target encoding from scratch.
> 2. Implement hash encoding (the "hashing trick") for a high-cardinality column.
> 3. Build a custom encoder that combines one-hot for low-cardinality and target
>    encoding for high-cardinality columns.
> 4. Compare an embedding learned by a small NN against target encoding.
> 5. Implement a Bayesian target encoding with a Beta prior.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. When is one-hot encoding a poor choice?
> 2. Why is target encoding leakage-prone, and how do you fix it?
> 3. What is the hashing trick and when does it help?
> 4. How do you encode a categorical column with 10,000 levels?
> 5. What is the difference between ordinal and label encoding?
> 6. How do tree-based models interact with target-encoded features?
> 7. What is an entity embedding and what model learns it?
> 8. How would you handle a categorical feature that has new levels at inference time?
