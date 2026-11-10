# Code Walkthrough — `05_visualization.ipynb`

> Module: **01-python-for-ml** | Notebook: **05_visualization.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 31
- **Markdown cells:** 25
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

> # 05 — Data Visualization
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
> ## 1. Concept Overview — Visualization — seeing before modelling

> A chart will tell you in two seconds what a `.describe()` will not: the shape
> of the distribution, the presence of outliers, the structure of clusters,
> the breakdown by class. Every model you build in the next 14 modules starts
> with a plot.
> 
> **Beginner track:** master the histogram, scatter, and boxplot — these are
> enough for 80% of exploratory work.
> **Advanced track:** add the residual plot, calibration curve, and
> confusion matrix — the diagnostic plots that come *after* a model is fit.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Plots are projections of high-dimensional data to 2-D. Three ideas govern how
> much information survives the projection:
> 
> - **Position** — most accurate encoding (Cleveland & McGill).
> - **Color hue** — good for categorical groups, bad for ordered quantities.
> - **Area / length** — good for magnitudes, but the eye underestimates area.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** An ASCII scatter plot in pure Python — the kind of
> thing you can use in a raw SSH session with no display.
> 
> **Why it matters:** the algorithm (project to a grid, place a `*` at each
> data point) is *exactly* what matplotlib does internally — just with pixels
> instead of characters. Understanding this demystifies all of plotting.
> 
> **What to watch for:** the `+ 1e-9` in the denominator is a tiny epsilon to
> avoid divide-by-zero when all x values are equal.


### Code cell #1
```python
# A scatter plot in pure Python (ASCII) — to appreciate what matplotlib gives us.
import random

def ascii_scatter(xs, ys, width=60, height=20):
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    grid = [[" "] * width for _ in range(height)]  # blank canvas
    for x, y in zip(xs, ys):
        # Map data coords to grid coords. +1e-9 prevents divide-by-zero.
        col = int((x - x_lo) / (x_hi - x_lo + 1e-9) * (width - 1))
        # Row 0 is top of plot, so invert: (height-1) - ...
        row = (height - 1) - int((y - y_lo) / (y_hi - y_lo + 1e-9) * (height - 1))
        grid[row][col] = "*"
    return "\n".join("".join(r) for r in grid)

rng = random.Random(0)
xs = [rng.gauss(0, 1) for _ in range(80)]
ys = [0.7 * x + rng.gauss(0, 0.4) for x in xs]   # correlated + noise
print(ascii_scatter(xs, ys))
```

**What this code does:**

- **Headline:** `import random`
- **Defines:** def `ascii_scatter`
- **Key imports:** `import random`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Four panels of the most common matplotlib plots on the
> same synthetic dataset: scatter, histogram, boxplot, and ECDF.
> 
> **Why it matters:** these four cover the vast majority of exploratory
> plots. Knowing them cold means you can characterise any 1-D or 2-D dataset
> in under a minute.
> 
> **What to watch for:** `plt.subplots(2, 2)` returns a 2-D array of axes —
> index with `axes[i, j]`. `plt.subplots(1, 4)` returns a 1-D array — index
> with `axes[i]`. The shape of `axes` matches the shape you passed.


### Code cell #2
```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.standard_normal(500)
y = 0.8 * x + 0.3 * rng.standard_normal(500)   # correlated with noise

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: scatter — relationship between two continuous variables.
axes[0, 0].scatter(x, y, alpha=0.5, edgecolor="white")
axes[0, 0].set_title("Scatter")

# Top-right: histogram — marginal distribution of one variable.
axes[0, 1].hist(y, bins=30, color="steelblue", edgecolor="white")
axes[0, 1].set_title("Histogram of y")

# Bottom-left: boxplot — five-number summary (min, q1, median, q3, max).
axes[1, 0].boxplot([x, y], labels=["x", "y"])
axes[1, 0].set_title("Boxplot")

# Bottom-right: ECDF — the most honest distribution plot.
axes[1, 1].plot(np.sort(x), np.linspace(0, 1, len(x)))
axes[1, 1].set_title("ECDF of x")

plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `import matplotlib.pyplot as plt`

> **What this does:** A `seaborn.pairplot` of iris — the *single most
> informative plot* in all of ML. It shows every pairwise scatter plus every
> diagonal histogram, colour-coded by class.
> 
> **Why it matters:** pairplot lets you see linear separability, clusters,
> correlations, and outliers all at once. It's the first command a senior ML
> engineer runs on a new tabular dataset.
> 
> **What to watch for:** `corner=True` only plots the lower triangle — half
> the cells, same information (the upper triangle is the mirror image).


### Code cell #3
```python
import seaborn as sns
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
sns.pairplot(df, hue="species", corner=True)
plt.show()
```

**What this code does:**

- **Headline:** `import seaborn as sns`
- **Key imports:** `import seaborn as sns`, `import pandas as pd`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** A `FacetGrid` on the seaborn `tips` dataset — small
> multiples of the same scatter, one per `time` × `sex` combination, coloured
> by smoker.
> 
> **Why it matters:** small multiples (Tufte) are the cleanest way to compare
> groups — same axes, same scale, only the subset changes. Once you learn
> `FacetGrid`, you stop writing for-loops of plots.
> 
> **What to watch for:** `map_dataframe` (not `map`) is the modern API that
> passes keyword arguments to the plotting function correctly.


### Code cell #4
```python
tips = sns.load_dataset("tips")
print(tips.head())
print("shape:", tips.shape)

# Small multiples: one panel per (time, sex), colour by smoker.
g = sns.FacetGrid(tips, col="time", row="sex", hue="smoker", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.add_legend()
plt.show()
```

**What this code does:**

- **Headline:** `tips = sns.load_dataset("tips")`

> ---
> 
> ## 6. Visualization — Visualization

> We use this section to demonstrate three plots that show up again and again
> in ML reports: the correlation heatmap, the residual plot, and the
> confusion matrix.

> **What this does:** Three diagnostic plots side-by-side: correlation
> heatmap, residual plot from a linear model, and confusion matrix from a
> decision tree. These are the three you'll be asked to produce in any
> model review.
> 
> **Why it matters:** the heatmap catches multicollinearity, the residual
> plot catches non-linearity and heteroscedasticity, the confusion matrix
> catches class-conditional errors. One of each per model = a basic model card.
> 
> **What to watch for:** the residual plot has a horizontal red line at 0.
> If residuals fan out (cone shape) as predictions grow, you have
> heteroscedasticity — switch to a log-transformed target or use robust SEs.


### Code cell #5
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. correlation heatmap — catches redundant features.
corr = tips.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Tips correlation")

# 2. residual plot — catches non-linearity and heteroscedasticity.
X = tips[["total_bill"]].to_numpy()
y = tips["tip"].to_numpy()
reg = LinearRegression().fit(X, y)
residuals = y - reg.predict(X)
axes[1].scatter(reg.predict(X), residuals, alpha=0.5)
axes[1].axhline(0, color="red", linestyle="--")   # the zero line
axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual")
axes[1].set_title("Residual plot")

# 3. confusion matrix — catches which classes are confused with which.
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
X_tr, X_te, y_tr, y_te = train_test_split(
    iris.drop(columns="species"), iris["species"], test_size=0.3, random_state=0)
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
cm = confusion_matrix(y_te, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_,
            yticklabels=clf.classes_, cmap="Blues", ax=axes[2])
axes[2].set_xlabel("predicted"); axes[2].set_ylabel("true")
axes[2].set_title("Iris confusion matrix")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.linear_model import LinearRegression`
- **Key imports:** `from sklearn.linear_model import LinearRegression`, `from sklearn.metrics import confusion_matrix`, `from sklearn.tree import DecisionTreeClassifier`, `from sklearn.model_selection import train_test_split`
- **Notable calls:** `LinearRegression`, `DecisionTree`, `train_test_split`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Three chart types you should *always* be able to produce for a model:
> 
> - **ROC / PR curve** — for binary classifiers.
> - **Residual plot** — for regression models.
> - **Confusion matrix** — for multi-class classifiers.
> 
> Each one catches a different family of failure modes.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Take the same scatter plot and try four color palettes — which conveys
>    the cluster structure best?
> 2. Plot a 4-D dataset using a 2-D scatter + size + color. Is it readable?
> 3. Make a 3-D plot with `mplot3d` — when does 3-D help and when does it hurt?

> **What this does:** Builds an interactive `altair` scatter with tooltips
> for the tips dataset. In Colab/JupyterLab this renders as a hoverable chart.
> 
> **Why it matters:** once you have more than a few hundred points, static
> plots stop being readable. Interactive tooltips let you drill into any
> point — essential for outlier investigation.
> 
> **What to watch for:** `altair` works in JupyterLab and Colab but not in
> the bare `jupyter notebook` classic. The `try/except` guards the import.


### Code cell #6
```python
# Interactive altair example (works in Colab / JupyterLab).
try:
    import altair as alt
    chart = (alt.Chart(tips).mark_circle().encode(
        x="total_bill", y="tip", color="sex", size="size",
        tooltip=["total_bill", "tip", "sex", "size"]
    ).properties(width=500, height=300))
    chart.display()
except ImportError:
    print("altair not installed — pip install altair")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import altair as alt`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Build a function `plot_distributions(df, target)` that draws a per-class
>    KDE for every numeric column in `df`.
> 2. Write `plot_feature_importances(model, feature_names)`.
> 3. Create a 4-panel diagnostic plot for a regression model (residuals, QQ,
>    scale-location, leverage).
> 4. Implement `plot_confusion_matrix(cm, labels)` from scratch with matplotlib.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Which visual encoding is the most accurate for quantitative values, and why?
> 2. When is a bar chart better than a pie chart — and when is it not?
> 3. How would you visualise a 100-dimensional dataset for a stakeholder?
> 4. What does a residual-vs-fitted plot tell you that an $R^2$ does not?
> 5. When would you use a log scale on the y-axis of a histogram?
> 6. What are the four most informative plots for evaluating a binary classifier?
> 7. How do you avoid misleading axes when presenting model performance?
> 8. What is the difference between `plt.show()` and the object-oriented API?
