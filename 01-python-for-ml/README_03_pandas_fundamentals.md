# Code Walkthrough — `03_pandas_fundamentals.ipynb`

> Module: **01-python-for-ml** | Notebook: **03_pandas_fundamentals.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 34
- **Markdown cells:** 26
- **Code cells:** 8

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

> # 03 — Pandas Fundamentals
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
> ## 1. Concept Overview — Pandas — labelled tabular data

> Pandas wraps the NumPy array with row/column labels, turning a bare matrix
> into a table you can query, join, group, reshape, and plot. It is the workhorse
> for everything that happens *before* model training: loading, cleaning,
> feature construction, and exploration.
> 
> **Beginner track:** learn the 5 verbs (`read`, `select`, `filter`, `groupby`,
> `merge`) — they cover 80% of real pandas code.
> **Advanced track:** also learn `transform`, `pivot`/`melt`, `query`, and
> categorical dtypes for memory efficiency.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Pandas doesn't add new math — it adds *structure* on top of NumPy:
> 
> - A `DataFrame` is a dict of `Series`, each backed by an array.
> - A `groupby` is the `GROUP BY` of relational algebra — a partition $\{G_g\}_{g\in G}$.
> - An aggregation is $f: 2^D \to \mathbb{R}$ applied per group.
> - A join is $R_1 \bowtie_{k} R_2 = \{(r_1, r_2) : r_1.k = r_2.k\}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** A 30-line reimplementation of a DataFrame with `filter`
> and `groupby_mean` — enough to demystify what pandas does under the hood.
> 
> **Why it matters:** Once you see a DataFrame is just a dict-of-columns plus
> an index, the pandas API stops feeling magical — and you start predicting
> what method names will exist (`transform`, `agg`, `apply` all become obvious).
> 
> **What to watch for:** real pandas uses *alignment on the index*. Our
> MiniDataFrame ignores the index — that's the difference between a toy and
> the real thing.


### Code cell #1
```python
# A tiny DataFrame implemented on top of dicts — to see what pandas gives us.
class MiniDataFrame:
    def __init__(self, columns):
        self.columns = dict(columns)  # name -> list  (a copy)

    def __getitem__(self, key):
        return self.columns[key]   # column lookup by name

    def filter(self, mask):
        """Keep only rows where mask is True — equivalent to df[mask]."""
        return MiniDataFrame({
            k: [v for v, m in zip(col, mask) if m]
            for k, col in self.columns.items()
        })

    def groupby_mean(self, by, value):
        """GROUP BY <by> and compute mean of <value> — relational algebra."""
        groups = {}
        for g, v in zip(self.columns[by], self.columns[value]):
            groups.setdefault(g, []).append(v)
        return {g: sum(vs) / len(vs) for g, vs in groups.items()}

    def __len__(self):
        return len(next(iter(self.columns.values())))  # any column has the answer

df = MiniDataFrame({
    "species": ["setosa", "versicolor", "setosa", "virginica", "setosa"],
    "petal_length": [1.4, 4.7, 1.5, 5.1, 1.3],
})
print("setosa rows:", len(df.filter([s == "setosa" for s in df["species"]])))
print("mean petal_length by species:", df.groupby_mean("species", "petal_length"))
```

**What this code does:**

- **Headline:** `class MiniDataFrame:`
- **Defines:** class `MiniDataFrame`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Loads the iris CSV with `pd.read_csv`, shows `head`,
> `dtypes`, and `describe` — the three calls you make on *every* new dataset.
> 
> **Why it matters:** `describe()` gives you the 5-number summary in one
> line. If the `min`, `max`, or `std` of any column looks wrong, you have a
> data quality issue — fix it before modelling.
> 
> **What to watch for:** `dtypes` shows you `object` for any string column.
> If a *numeric* column shows as `object`, it has mixed types or strings —
> convert it with `pd.to_numeric(df[col], errors=


### Code cell #2
```python
import pandas as pd

csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(csv_url)               # the pandas one-liner that does all the work
print("shape:", df.shape)               # (rows, cols)
print(df.head())                         # first 5 rows — sanity check
print("\ndtypes:\n", df.dtypes)         # column types — watch for "object" in numeric cols
print("\ndescribe():\n", df.describe()) # 5-number summary per numeric column
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`

> **What this does:** A `groupby` + `agg` + `sort_values` chain — the most
> common pandas pattern for feature summarisation.
> 
> **Why it matters:** Once you can read this fluently, you can do 80% of
> real-world feature engineering. The chain pattern (each method returns a
> new DataFrame) is how modern pandas code is written.
> 
> **What to watch for:** the named-aggregation syntax (`mean_petal=("petal_length", "mean")`)
> is the cleanest way to name your output columns. Don't use a dict of dicts
> — that's the deprecated pre-0.25 syntax.


### Code cell #3
```python
# Group-by, aggregation, sorting — the heart of feature engineering.
# Pattern: df.groupby(key).agg(out_col=(in_col, func), ...).sort_values(by)
agg = (df.groupby("species")
         .agg(mean_petal=("petal_length", "mean"),
              max_petal=("petal_length", "max"),
              count=("petal_length", "size"))
         .sort_values("mean_petal", ascending=False))
print(agg)
```

**What this code does:**

- **Headline:** `agg = (df.groupby("species")`

> **What this does:** Demonstrates `merge` (relational join) and `pivot`
> (long → wide reshape) — two operations that show up in every feature
> engineering pipeline.
> 
> **Why it matters:** Real datasets are rarely tidy. You'll have a customer
> table and a transactions table that need joining; you'll have a long-format
> metrics table that needs to become wide for modelling.
> 
> **What to watch for:** `how="inner"` keeps only matching keys; `how="outer"`
> keeps everything and fills non-matches with `NaN`. Choose based on whether
> missing data is informative for your model.


### Code cell #4
```python
# Joins and reshaping — relational algebra and tidy data.
left = pd.DataFrame({"id": [1, 2, 3], "feature_a": [0.1, 0.5, 0.9]})
right = pd.DataFrame({"id": [2, 3, 4], "feature_b": [10, 20, 30]})

inner = left.merge(right, on="id", how="inner")   # only rows present in BOTH
outer = left.merge(right, on="id", how="outer")   # every row, NaN where missing
print("inner join:\n", inner)
print("\nouter join:\n", outer)

# pivot: long → wide. melt: wide → long.
long = pd.DataFrame({"id": [1, 1, 2, 2],
                     "metric": ["a", "b", "a", "b"],
                     "value": [10, 20, 30, 40]})
wide = long.pivot(index="id", columns="metric", values="value")
print("\nwide:\n", wide)
```

**What this code does:**

- **Headline:** `left = pd.DataFrame({"id": [1, 2, 3], "feature_a": [0.1, 0.5, 0.9]})`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the Titanic dataset and computes missingness per
> column plus a pivot table of survival rate by `Sex` × `Pclass`.
> 
> **Why it matters:** Titanic is the *hello world* of messy data — it has
> missing ages, embarked ports, and cabins, plus a strong class imbalance.
> It teaches you every preprocessing decision before you touch a model.
> 
> **What to watch for:** `isna().sum()` is the most common one-liner in data
> science. The `mean()` of a boolean series gives the *fraction* of True — so
> `isna().mean()` gives the missing rate directly.


### Code cell #5
```python
# The Titanic dataset — messy, real-world-ish.
titanic = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
print("shape:", titanic.shape)
print(titanic.isna().sum())  # missingness per column
print("\n survival rate by sex and class:")
print(titanic.pivot_table(values="Survived", index="Sex",
                         columns="Pclass", aggfunc="mean"))
```

**What this code does:**

- **Headline:** `titanic = pd.read_csv(`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Two panels: a histogram of petal length by species (with
> KDE overlay) and a correlation heatmap of all numeric features.
> 
> **Why it matters:** The KDE histogram tells you whether classes are
> *separable* in a feature; the heatmap tells you whether features are
> *redundant* (high correlation = drop one).
> 
> **What to watch for:** correlation is *linear* association. Two features can
> be perfectly dependent yet have zero Pearson correlation — always plot
> the scatter too.


### Code cell #6
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: per-class distribution. kde=True overlays a smooth density estimate.
sns.histplot(data=df, x="petal_length", hue="species", kde=True, ax=axes[0])
axes[0].set_title("Petal length by species")

# Right: correlation heatmap. center=0 means negative=blue, positive=red.
corr = df.drop(columns="species").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title("Iris feature correlation")
plt.tight_layout()
plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For tabular data, before training any model, we evaluate *data quality*:
> 
> - Missingness rate per column.
> - Class balance of the target.
> - Train/test leakage via duplicated rows.

> **What this does:** Three one-liner data-quality checks on Titanic:
> missingness rate, class balance, and duplicate rows.
> 
> **Why it matters:** If you skip this step, every downstream result is
> suspect — high missingness tells you which columns to impute or drop,
> imbalance tells you which metric to use, and duplicate rows would cause
> train/test leakage.
> 
> **What to watch for:** `value_counts(normalize=True)` gives fractions, not
> raw counts — perfect for spotting imbalances like 99% / 1%.


### Code cell #7
```python
missing = titanic.isna().mean().sort_values(ascending=False)
print("missing rate per column:\n", (missing * 100).round(2))

print("\nclass balance of target:")
print(titanic["Survived"].value_counts(normalize=True))  # fractions, not counts

print("\nduplicate rows:", titanic.duplicated().sum())
```

**What this code does:**

- **Headline:** `missing = titanic.isna().mean().sort_values(ascending=False)`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Impute missing `Age` values with the median grouped by `Pclass` — compare to a global median.
> 2. One-hot encode `Embarked` and `Sex` with `pd.get_dummies`.
> 3. Bucket `Age` into 5 quantile bins with `pd.qcut`.

> **What this does:** Three feature engineering steps chained together:
> group-wise median imputation, one-hot encoding, and quantile bucketing.
> 
> **Why it matters:** These three patterns — *impute*, *encode*, *bin* —
> cover most of what feature engineering *is* in tabular ML. Combined with
> scaling (Module 03), you have 90% of preprocessing.
> 
> **What to watch for:** `groupby("Pclass")["Age"].transform(...)` keeps the
> original row index — a plain `groupby().median()` would return a smaller
> DataFrame indexed by `Pclass` and would *not* align with the original rows.


### Code cell #8
```python
titanic2 = titanic.copy()

# Impute Age with the median of each Pclass group (better than global median).
# transform() returns a Series aligned to the original index — critical for filling.
titanic2["Age"] = titanic2.groupby("Pclass")["Age"].transform(
    lambda s: s.fillna(s.median())
)
print("remaining nulls:", titanic2["Age"].isna().sum())

# One-hot encode two categorical columns. drop_first=True avoids the
# dummy-variable trap (perfect multicollinearity) for linear models.
titanic2 = pd.get_dummies(titanic2, columns=["Sex", "Embarked"], drop_first=True)
print("new columns:", [c for c in titanic2.columns if "Sex_" in c or "Embarked_" in c])

# Bucket Age into 5 equal-frequency bins. Quantile bins are robust to outliers.
titanic2["age_bin"] = pd.qcut(titanic2["Age"], q=5, labels=["q1", "q2", "q3", "q4", "q5"])
print(titanic2["age_bin"].value_counts())
```

**What this code does:**

- **Headline:** `titanic2 = titanic.copy()`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Load two CSVs that share a key, perform an outer join, and report rows that survived.
> 2. Build a function `report_missing(df)` that returns a styled summary table.
> 3. Pivot a long-format sales table to wide-format, then melt it back.
> 4. Compute rolling 7-day mean of a time-indexed series.
> 5. Detect rows where any numeric column is more than 3 standard deviations from its mean.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. `loc` vs `iloc` — when does each matter, and what is the common bug?
> 2. What is the difference between `merge` and `join`?
> 3. What is a `SettingWithCopyWarning` and how do you avoid it?
> 4. Why does pandas have both `map`, `apply`, and `applymap` (now `map`)?
> 5. How would you handle a 50 GB CSV that does not fit in memory?
> 6. Explain the difference between `groupby().mean()` and `groupby().transform("mean")`.
> 7. What is `pivot_table` doing under the hood?
> 8. How do categorical dtypes save memory, and when are they a bad idea?
