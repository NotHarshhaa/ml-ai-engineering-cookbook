# Code Walkthrough — `04_statistics.ipynb`

> Module: **01-python-for-ml** | Notebook: **04_statistics.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 30
- **Markdown cells:** 24
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

> # 04 — Statistics for ML
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
> ## 1. Concept Overview — Statistics — the language of uncertainty

> Machine learning is fundamentally about reasoning under uncertainty. Statistics
> gives us the tools to *describe* distributions, *estimate* parameters, and
> *test* hypotheses — all of which we lean on when evaluating models.
> 
> **Beginner track:** focus on mean, variance, and the t-test — these will
> let you read any ML evaluation table.
> **Advanced track:** add bootstrap CIs, permutation tests, and effect sizes
> to your reporting — these are what separate a junior from a senior result.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> We use four families of ideas throughout ML:
> 
> - **Descriptive** — mean $\bar{x}$, variance $\sigma^2$, quantiles.
> - **Probability** — PMF/PDF, CDF, joint and conditional distributions.
> - **Estimation** — MLE, method of moments, confidence intervals.
> - **Hypothesis testing** — $H_0$, $H_1$, $p$-values, $t$-tests, $\chi^2$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Hand-rolls `mean`, `variance`, `std`, and `quantile`
> from a list of numbers — the four descriptive statistics you will use every
> day.
> 
> **Why it matters:** Implementing these once demystifies them. The variance
> is just "the average of (x - mean)^2", not a magic library call. The
> quantile is just linear interpolation on the sorted data.
> 
> **What to watch for:** `ddof=1` (delta-degrees-of-freedom) is the difference
> between the *sample* variance (divide by $n-1$, unbiased) and the
> *population* variance (divide by $n$). Almost always use `ddof=1`.


### Code cell #1
```python
import random, math

def mean(xs): return sum(xs) / len(xs)

def variance(xs, ddof=1):
    """Sample variance with ddof=1 (unbiased). Population variance uses ddof=0."""
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)

def std(xs, ddof=1): return math.sqrt(variance(xs, ddof))

def quantile(xs, q):
    """Linear-interpolation quantile — matches numpy.percentile default."""
    s = sorted(xs)
    pos = q * (len(s) - 1)              # position in [0, len-1]
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)   # linear interpolation

data = [random.gauss(5, 2) for _ in range(1000)]
print(f"mean={mean(data):.3f} variance={variance(data):.3f} std={std(data):.3f}")
print(f"q25={quantile(data, 0.25):.3f} median={quantile(data, 0.5):.3f} q75={quantile(data, 0.75):.3f}")
```

**What this code does:**

- **Headline:** `import random, math`
- **Defines:** def `mean` — Sample variance with ddof=1 (unbiased). Population variance uses ddof=0.; def `variance` — Sample variance with ddof=1 (unbiased). Population variance uses ddof=0.; def `std` — Linear-interpolation quantile — matches numpy.percentile default.; def `quantile` — Linear-interpolation quantile — matches numpy.percentile default.
- **Key imports:** `import random`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** A two-sample Welch t-test (unequal variances) plus a
> 95% confidence interval for the mean of one sample, using `scipy.stats`.
> 
> **Why it matters:** The t-test answers "are these two means really
> different, or could the difference be noise?" The CI answers "in what
> range would the true mean plausibly lie?" Together they cover most of
> ML model comparison.
> 
> **What to watch for:** use `equal_var=False` (Welch) by default — the
> classic Student's t-test assumes equal variances, which is almost never
> true in real data.


### Code cell #2
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)
a = rng.normal(loc=5, scale=2, size=200)
b = rng.normal(loc=5.6, scale=2, size=200)   # true mean differs by 0.6

print("mean a/b:", a.mean(), b.mean())
print("std  a/b:", a.std(ddof=1), b.std(ddof=1))

# Two-sample t-test: is the mean difference statistically significant?
t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)  # Welch = safer default
print(f"Welch t = {t_stat:.3f}  p = {p_value:.4f}")

# 95% confidence interval for the mean of `a`.
# sem = standard error of the mean = std / sqrt(n).
ci = stats.t.interval(0.95, df=len(a)-1, loc=a.mean(), scale=stats.sem(a))
print("95% CI for mean(a):", tuple(round(x, 3) for x in ci))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `from scipy import stats`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Computes per-species summary stats on iris, and runs a
> one-sample t-test on setosa petal length vs the hypothesised mean of 1.5.
> 
> **Why it matters:** this is the exact pattern you'd use to compare model
> per-species performance — `groupby` + `agg`, then a t-test against a
> baseline.
> 
> **What to watch for:** the p-value here will be very small (the true mean
> of setosa petal length is around 1.46, slightly below 1.5) — but small p ≠
> big effect. Always look at the effect size too.


### Code cell #3
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Per-species summary stats — the standard "report by group" pattern.
print(df.groupby("species").agg(
    mean_pl=("petal_length", "mean"),
    std_pl=("petal_length", "std"),
    median_pl=("petal_length", "median"),
))

# Is mean petal length of setosa significantly different from 1.5?
setosa = df.loc[df.species == "setosa", "petal_length"]
t, p = stats.ttest_1samp(setosa, popmean=1.5)
print(f"one-sample t-test vs 1.5: t={t:.3f} p={p:.4f}")
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Boxplot and KDE of petal length by species — the two
> complementary views of a grouped distribution.
> 
> **Why it matters:** boxplots show quartiles + outliers; KDEs show the
> *shape* of the distribution (unimodal vs bimodal, skew). Use both — they
> answer different questions.
> 
> **What to watch for:** a KDE will *smooth over* a bimodal distribution if
> the bandwidth is too large. Always check with a histogram first.


### Code cell #4
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Boxplot: quartiles + whiskers + outlier dots. Best for comparing groups.
sns.boxplot(data=df, x="species", y="petal_length", ax=axes[0])
axes[0].set_title("Petal length by species (boxplot)")

# KDE: smoothed density. Best for seeing distribution shape.
sns.kdeplot(data=df, x="petal_length", hue="species", fill=True, ax=axes[1])
axes[1].set_title("Petal length density (KDE)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import seaborn as sns`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Statistical evaluation of a classifier:
> 
> - **Confidence interval on accuracy** — binomial proportion CI.
> - **McNemar's test** — comparing two classifiers on the same test set.
> - **Cohen's kappa** — agreement beyond chance.

> **What this does:** Computes a Wilson 95% CI for a 90/100 accuracy, and
> Cohen's kappa for two toy classifiers against ground truth.
> 
> **Why it matters:** "90% accuracy" with a CI of [82%, 95%] is a very
> different statement from "90% accuracy" with a CI of [89%, 91%]. The
> first is a noisy estimate, the second is precise. Report the CI, not just
> the point estimate.
> 
> **What to watch for:** Cohen's kappa corrects for chance agreement — a
> kappa of 0 means your classifier is no better than guessing the majority
> class, even if raw accuracy is 99% on an imbalanced dataset.


### Code cell #5
```python
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import cohen_kappa_score

# Simulate: 90 correct out of 100 — what is the 95% CI for accuracy?
# Wilson method is recommended over the "normal approximation" for small n.
n, k = 100, 90
lo, hi = proportion_confint(count=k, nobs=n, alpha=0.05, method="wilson")
print(f"accuracy {k/n:.2f}, 95% CI [{lo:.3f}, {hi:.3f}]")

# Two toy classifiers — one perfect, one with two errors.
truth   = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
pred_a  = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]   # 2 errors
pred_b  = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]   # 0 errors (matches truth)
print("kappa A:", round(cohen_kappa_score(truth, pred_a), 3))
print("kappa B:", round(cohen_kappa_score(truth, pred_b), 3))
```

**What this code does:**

- **Headline:** `from statsmodels.stats.proportion import proportion_confint`
- **Key imports:** `from statsmodels.stats.proportion import proportion_confint`, `from sklearn.metrics import cohen_kappa_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Draw 1000 samples from a Normal(0,1) and plot the empirical CDF against the
>    theoretical CDF — confirm visually that they converge.
> 2. Run a $\chi^2$ test of independence on `titanic` `Sex` vs `Survived`.
> 3. Bootstrap a 95% confidence interval for the median of a skewed distribution.

> **What this does:** Plots the empirical CDF (ECDF) of 1000 Normal samples
> against the theoretical Normal CDF — visual proof of the convergence that
> underlies most of statistics. Then runs a $\chi^2$ test of independence on
> Titanic `Sex` × `Survived`.
> 
> **Why it matters:** The ECDF is the *nonparametric* estimator of a
> distribution — it makes no shape assumptions. Comparing it to a theoretical
> CDF is the most honest distribution check you can do.
> 
> **What to watch for:** the $\chi^2$ test on Titanic will return a *tiny*
> p-value — Sex and Survived are highly dependent. In practice that means
> Sex is a powerful predictor — a statistical insight that becomes a feature
> engineering insight.


### Code cell #6
```python
rng = np.random.default_rng(0)
samples = rng.standard_normal(1000)
xs = np.linspace(-4, 4, 200)

# ECDF: fraction of samples <= x.
ecdf = np.array([np.mean(samples <= x) for x in xs])
tcdf = stats.norm.cdf(xs)   # theoretical Normal(0,1) CDF

plt.figure(figsize=(6,4))
plt.plot(xs, ecdf, label="empirical CDF")
plt.plot(xs, tcdf, label="theoretical CDF", linestyle="--")
plt.legend(); plt.title("ECDF vs Normal CDF"); plt.show()

# Chi^2 test of independence: are Sex and Survived independent?
# H0: independent.  Small p -> reject H0 -> they are dependent.
titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
table = pd.crosstab(titanic["Sex"], titanic["Survived"])   # contingency table
chi2, p, dof, _ = stats.chi2_contingency(table)
print(f"chi2 = {chi2:.2f}  p = {p:.3e}  dof = {dof}")
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(0)`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a `bootstrap_ci(sample, statistic, n_boot=1000, ci=0.95)` from scratch.
> 2. Implement a permutation test for the difference of two means.
> 3. Implement Cohen's $d$ effect size.
> 4. Compute Spearman rank correlation between two features and compare to Pearson.
> 5. Implement an MLE estimator for the $\lambda$ of an Exponential distribution.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between sample mean and population mean?
> 2. When should you use a $t$-test instead of a $z$-test?
> 3. Explain what a $p$-value is — and what it is *not*.
> 4. What is the central limit theorem and why does it matter for ML evaluation?
> 5. What does it mean for an estimator to be *unbiased* vs *consistent*?
> 6. Why is correlation not causation, and how does this affect feature selection?
> 7. What is the bootstrap, and when is it preferable to a parametric test?
> 8. How would you test whether two classifiers are significantly different?
