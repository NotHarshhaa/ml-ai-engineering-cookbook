# Code Walkthrough — `01_data_preprocessing.ipynb`

> Module: **03-feature-engineering** | Notebook: **01_data_preprocessing.ipynb**
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

> # 01 — Data Preprocessing
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
> ## 1. Concept Overview — Data preprocessing — making raw data model-ready

> Real-world data is messy: missing values, mixed types, outliers, and
> inconsistent scales. **Preprocessing** is the set of transformations that
> convert raw data into a form where a model can learn from it.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Three preprocessing operations have a precise mathematical meaning:
> 
> - **Imputation**: replace $x_{ij} = \text{NA}$ with $\hat{x}_{ij}$ — typically
>   the mean $\bar{x}_j$, median, or a model-based estimate.
> - **Outlier clipping**: $\tilde{x} = \min(\max(x, q_\text{low}), q_\text{high})$.
> - **Encoding**: map a categorical value $c \in C$ to a vector $\phi(c) \in \mathbb{R}^k$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "age":     rng.integers(18, 70, 20).astype(float),
    "income":  rng.normal(50000, 15000, 20).round(2),
    "score":   rng.uniform(0, 100, 20).round(1),
    "city":    rng.choice(["NY", "LA", "SF"], 20),
})
# Inject missing values and one outlier.
df.loc[df.sample(4, random_state=1).index, "age"]    = np.nan
df.loc[df.sample(3, random_state=2).index, "income"] = np.nan
df.loc[5, "income"] = 500000  # outlier
df.head(8)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `import pandas as pd`


### Code cell #2
```python
# Mean / median imputation from scratch.
def impute_mean(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out[c].mean())
    return out

def impute_group_median(df, target, group):
    out = df.copy()
    out[target] = out.groupby(group)[target].transform(
        lambda s: s.fillna(s.median())
    )
    return out

def clip_outliers(df, cols, k=1.5):
    out = df.copy()
    for c in cols:
        q1, q3 = out[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k*iqr, q3 + k*iqr
        out[c] = out[c].clip(lo, hi)
    return out

clean = clip_outliers(impute_group_median(df, "income", "city"),
                      ["age", "income", "score"], k=1.5)
clean = impute_mean(clean, ["age"])
clean.head(8)
```

**What this code does:**

- **Headline:** `def impute_mean(df, cols):`
- **Defines:** def `impute_mean`; def `impute_group_median`; def `clip_outliers`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #3
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric = ["age", "income", "score"]
categorical = ["city"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", KNNImputer(n_neighbors=3)),
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

X_proc = preprocessor.fit_transform(df)
print("processed shape:", X_proc.shape)
print("feature names:", preprocessor.get_feature_names_out())
```

**What this code does:**

- **Headline:** `from sklearn.impute import SimpleImputer, KNNImputer`
- **Key imports:** `from sklearn.impute import SimpleImputer`, `from sklearn.preprocessing import OneHotEncoder`, `from sklearn.compose import ColumnTransformer`, `from sklearn.pipeline import Pipeline`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #4
```python
titanic = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
print("missing rate per column:")
print((titanic.isna().mean() * 100).round(2).sort_values(ascending=False))
print("\nnumeric/categorical split:")
print(titanic.dtypes.groupby(titanic.dtypes).size())
```

**What this code does:**

- **Headline:** `titanic = pd.read_csv(`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #5
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=titanic[["Age", "Fare"]], ax=axes[0])
axes[0].set_title("Boxplots: visible outliers in Fare")

sns.heatmap(titanic.isna(), cbar=False, ax=axes[1])
axes[1].set_title("Missing-value map (yellow = NA)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> We measure preprocessing indirectly, by its effect on a downstream model.
> A common protocol: hold the model fixed, vary the preprocessing, watch
> cross-validated accuracy.


### Code cell #6
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

X = titanic[["Pclass", "Age", "Fare", "SibSp", "Parch"]].copy()
y = titanic["Survived"]

# Baseline: drop rows with NA, no scaling.
X_drop = X.dropna(); y_drop = y.loc[X_drop.index]
acc_drop = cross_val_score(LogisticRegression(max_iter=1000), X_drop, y_drop,
                            cv=5, scoring="accuracy").mean()

# Preprocessed: impute + scale.
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000)),
])
acc_pipe = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()

print(f"drop-NA accuracy:  {acc_drop:.3f}  (n={len(X_drop)})")
print(f"impute+scale:     {acc_pipe:.3f}  (n={len(X)})")
```

**What this code does:**

- **Headline:** `from sklearn.linear_model import LogisticRegression`
- **Key imports:** `from sklearn.linear_model import LogisticRegression`, `from sklearn.model_selection import cross_val_score`, `from sklearn.metrics import accuracy_score`, `from sklearn.preprocessing import StandardScaler`
- **Notable calls:** `LogisticRegression`, `cross_val_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace median imputation with `KNNImputer(n_neighbors=5)` — does accuracy change?
> 2. Cap `Fare` at its 99th percentile instead of leaving outliers — does it help?
> 3. Add an `is_missing` indicator column for `Age` — does the model exploit it?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement `report_missing(df)` returning a tidy DataFrame with missing
>    count, rate, and a `drop?` recommendation per column.
> 2. Implement `IQR_clipping(X)` as a sklearn `TransformerMixin`.
> 3. Build a preprocessing `Pipeline` for a 50-column mixed dataset.
> 4. Implement MICE-style imputation (iterative) using scikit-learn's
>    `IterativeImputer`.
> 5. Write a function that detects and logs every preprocessing decision for auditability.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. When is mean imputation a bad idea? Give two examples.
> 2. What is the difference between mean, median, and KNN imputation?
> 3. Why should preprocessing be fit only on the training set?
> 4. What is data leakage, and how does preprocessing cause it?
> 5. What is an `is_missing` indicator and when does it help?
> 6. How would you preprocess a column with 80% missing values?
> 7. Why use `ColumnTransformer` instead of applying the same imputer to every column?
> 8. How would you handle categorical columns with new values at inference time?
