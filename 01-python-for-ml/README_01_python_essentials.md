# Code Walkthrough — `01_python_essentials.ipynb`

> Module: **01-python-for-ml** | Notebook: **01_python_essentials.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 38
- **Markdown cells:** 30
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

> # 01 — Python Essentials for ML
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
> ## 1. Concept Overview — Python Essentials

> Python is the lingua franca of Machine Learning. Before we touch a single
> model, we need fluency in the core language features that show up in every
> ML codebase: data structures, comprehensions, functions, classes, and
> iteration patterns.
> 
> **Why this matters for ML:** most ML engineering bugs are not math bugs —
> they are Python bugs (mutation, aliasing, off-by-one indexing, slow loops).
> Mastering these primitives pays dividends across every later module.
> 
> **Beginner track:** read the prose + comments and run every cell.
> **Advanced track:** also time each block, profile it with `cProfile`, and
> ask *"what would happen if I made this 1,000× bigger?"*

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Python itself is not the math — but it provides the *vocabulary* we use to
> express math. We will lean on:
> 
> - Scalar arithmetic and broadcasting (covered with NumPy in the next notebook).
> - Vector and matrix indexing that mirrors linear-algebra notation.
> - Iteration as the computational realisation of $\sum$ and $\prod$.
> - Functions as composable transformations $f \circ g$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds the three data structures every ML script uses —
> an ordered list of samples, a `Counter` for class-label tallies, a
> `defaultdict(list)` for grouped buckets (e.g. label → list of feature vectors),
> and a `set` for vocabulary.
> 
> **Why it matters:** Most ML pipelines start with one of these three
> shapes: a sequence, a tally, or a grouping. Recognising which one your
> data takes is half the battle of writing clean code.
> 
> **What to watch for:** `Counter` and `defaultdict` are *not* exotic — they
> are standard library. Don't reinvent them with `try/except KeyError`.


### Code cell #1
```python
# ---- Data structures that every ML codebase uses ----
import random
from collections import Counter, defaultdict

# Lists: ordered, mutable — the backbone of sequence data.
# A list of random floats could represent a column of feature values.
samples = [random.random() for _ in range(10)]
print("min/max/len:", min(samples), max(samples), len(samples))

# Dicts: label -> count, feature -> weight, token -> id.
# Counter is the cleanest way to tally class labels.
labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
label_counts = Counter(labels)
print("label counts:", dict(label_counts))

# defaultdict(list): grouped buckets, e.g. label -> list of feature vectors.
# Avoids the "if key not in d: d[key] = []" boilerplate.
grouped = defaultdict(list)
for label, value in zip(labels, samples):
    grouped[label].append(value)
print("grouped:", dict(grouped))

# Sets: vocabulary, seen indices, deduplication.
# Membership test is O(1) on average — use this for fast lookups.
vocab = set(labels)
print("vocab:", vocab)
```

**What this code does:**

- **Headline:** `import random`
- **Key imports:** `import random`, `from collections import Counter`

> **Takeaway:** List = sequence, dict = keyed lookup, set = membership. Pick
> the structure that matches your *access pattern*, not the order you wrote
> the data in.

> **What this does:** Defines three small functions: a pure `mse` loss, a
> `compose` higher-order function that realises mathematical composition
> $(f \circ g)(x) = f(g(x))$, and a quick demonstration.
> 
> **Why it matters:** Loss functions in ML are *pure* (same inputs → same
> output, no side effects) — keeping yours pure makes them trivial to test
> and vectorise later. Higher-order functions (functions that take or return
> functions) are how you build composable ML libraries.
> 
> **What to watch for:** lambdas are convenient but their stack traces are
> anonymous — give real functions real names for anything you will debug.


### Code cell #2
```python
# ---- Functions, *args/**kwargs, and pure transforms ----
def mse(y_true, y_pred):
    """Mean squared error — a pure function, easy to test and vectorise."""
    n = len(y_true)                                            # sample count
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n  # 1/n Σ (yt-yp)^2

def compose(f, g):
    """Mathematical composition: (f . g)(x) = f(g(x))."""
    return lambda x: f(g(x))

# Two tiny lambdas we will compose.
square = lambda x: x * x
plus_one = lambda x: x + 1
square_then_plus_one = compose(plus_one, square)  # f(g(x)) = (x^2)+1

print("MSE([1,2,3], [1,2,4]):", mse([1, 2, 3], [1, 2, 4]))  # 1/3 * (0+0+1)
print("compose ->", square_then_plus_one(3))                # (3^2)+1 = 10
```

**What this code does:**

- **Headline:** `def mse(y_true, y_pred):`
- **Defines:** def `mse` — Mean squared error — a pure function, easy to test and vectorise.; def `compose` — Mathematical composition: (f . g)(x) = f(g(x)).
- **Notable calls:** `transforms`

> **Takeaway:** Keep your loss functions pure. Then you can swap `mse` for
> `mae`, `huber`, `cross_entropy` without touching the training loop.

> **What this does:** Implements one-variable linear regression as a class,
> using the closed-form OLS formula $\hat{\beta} = (X^\top X)^{-1} X^\top y$
> rearranged for the 1-D case as $\n\text{slope} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$.
> 
> **Why it matters:** OLS is the *first* model in every ML course for a
> reason — it's a one-liner, it has a closed form, and it's the baseline you
> compare every fancier model against.
> 
> **What to watch for:** when `den == 0` (a constant feature), the slope is
> undefined. Always guard against division by zero in your own code.


### Code cell #3
```python
# ---- Classes: a tiny Linear Regression scratch model ----
class LinearRegressionScratch:
    """One-variable linear regression fit with closed-form OLS."""
    def __init__(self):
        self.slope = 0.0      # β1
        self.intercept = 0.0  # β0

    def fit(self, x, y):
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        # Cov(x, y) = Σ (xi - x̄)(yi - ȳ)
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        # Var(x) = Σ (xi - x̄)^2
        den = sum((xi - x_mean) ** 2 for xi in x)
        self.slope = num / den if den else 0.0   # guard against constant feature
        self.intercept = y_mean - self.slope * x_mean  # β0 = ȳ - β1 * x̄

    def predict(self, x):
        # y_hat = β0 + β1 * x
        return [self.slope * xi + self.intercept for xi in x]

# Toy data: roughly y = 2x + 0
xs = [1, 2, 3, 4, 5]
ys = [2.1, 3.9, 6.1, 8.0, 9.9]
model = LinearRegressionScratch()
model.fit(xs, ys)
print(f"slope={model.slope:.3f} intercept={model.intercept:.3f}")
print("predictions:", model.predict([6, 7]))
```

**What this code does:**

- **Headline:** `class LinearRegressionScratch:`
- **Defines:** class `LinearRegressionScratch` — One-variable linear regression fit with closed-form OLS.
- **Notable calls:** `LinearRegression`

> **Takeaway:** A model is just (state + fit + predict). Every model you
> meet later in scikit-learn follows the same `.fit() / .predict()` shape —
> so this class is literally the same interface, hand-rolled.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Shows three standard-library tools ML engineers reach
> for daily: `itertools.powerset` for feature-subset search, `functools.reduce`
> for folding a sequence into a scalar (e.g. cumulative training loss), and
> `pathlib.Path` for sane dataset paths.
> 
> **Why it matters:** The standard library is *fast* (C-implemented) and has
> no extra dependencies. Knowing what's already there saves you from writing
> and maintaining your own buggy versions.
> 
> **What to watch for:** never build dataset paths with string concatenation
> (`dir + "/" + file`). It breaks on Windows, and it's where 50% of "works
> on my machine" bugs live.


### Code cell #4
```python
# Python standard library tools an ML engineer uses daily.
from itertools import chain, combinations, product
from functools import reduce
from pathlib import Path

# powerset — useful for feature-subset search.
# Returns an iterator over ALL 2^n subsets, so use only for small n.
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

print("feature subsets:", list(powerset(["age", "income", "score"])))

# reduce — fold a sequence into a scalar (e.g. cumulative loss).
# At each step we append the new value to the running total list.
losses = [0.9, 0.6, 0.4, 0.3]
running = reduce(lambda acc, x: acc + [acc[-1] + x], losses, [0])
print("cumulative loss:", running)

# pathlib — the sane way to handle dataset paths.
# "/" works on every OS; you can chain, glob, read_text() directly.
data_dir = Path("/content/datasets")  # Colab-friendly convention
print("csv path:", data_dir / "iris.csv")
```

**What this code does:**

- **Headline:** `from itertools import chain, combinations, product`
- **Defines:** def `powerset`
- **Key imports:** `from itertools import chain`, `from functools import reduce`, `from pathlib import Path`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads a CSV with the standard-library `csv` module
> (no pandas yet) and the `urllib.request.urlretrieve` helper, then counts
> rows and unique species.
> 
> **Why it matters:** Doing it once by hand shows you exactly what pandas'
> `read_csv` does for you — and where it can fail silently (mixed dtypes,
> quoting, encoding). When you do hit a weird CSV, you'll know how to drop
> down a level.
> 
> **What to watch for:** `tempfile.NamedTemporaryFile(delete=False)` keeps
> the file on disk after we close it; we explicitly `os.unlink` it at the
> end so we don't leak temp files across notebook runs.


### Code cell #5
```python
# Load a tiny CSV by hand, no pandas yet — to see what pandas will do for us.
import csv
from urllib.request import urlretrieve
import tempfile, os

csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
urlretrieve(csv_url, tmp.name)        # downloads to a local temp file

with open(tmp.name) as f:
    reader = csv.DictReader(f)        # first row becomes dict keys
    rows = list(reader)               # materialise to a list of dicts

print("columns:", list(rows[0].keys()))
print("first row:", rows[0])
print("n rows:", len(rows))
print("species set:", {r["species"] for r in rows})  # set comprehension
os.unlink(tmp.name)                   # clean up the temp file
```

**What this code does:**

- **Headline:** `import csv`
- **Key imports:** `import csv`, `from urllib.request import urlretrieve`, `import tempfile`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Implements a pure-Python text histogram (no matplotlib)
> so you can sanity-check a distribution in any environment — including a
> raw SSH session with no display.
> 
> **Why it matters:** Before you fire up matplotlib, *always* eyeball your
> data with a 5-second text histogram. You'd be amazed how often this catches
> a scale bug, a NaN infestation, or a units mismatch before any model is
> ever trained.
> 
> **What to watch for:** `int((v - lo) / step)` — the `min(...)` clamp on
> the next line prevents an `IndexError` for the largest value, which would
> otherwise land in a bin of index `bins` instead of `bins - 1`.


### Code cell #6
```python
# Pure-Python text-based histogram — quick sanity check before plotting.
def text_hist(values, bins=10, width=40):
    lo, hi = min(values), max(values)
    step = (hi - lo) / bins if hi > lo else 1
    counts = [0] * bins
    for v in values:
        # min() clamps the max value to the last bin (avoids off-by-one).
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    m = max(counts) or 1                    # avoid div-by-zero on empty data
    for i, c in enumerate(counts):
        bar = "#" * int(c / m * width)        # scale bar to fixed pixel width
        print(f"[{lo + i*step:6.2f}, {lo + (i+1)*step:6.2f}) {bar} ({c})")

import random
data = [random.gauss(0, 1) for _ in range(500)]
text_hist(data)
```

**What this code does:**

- **Headline:** `def text_hist(values, bins=10, width=40):`
- **Defines:** def `text_hist`
- **Key imports:** `import random`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> We evaluate Python-level code (not models yet) on three axes:
> 
> - **Correctness** — does the function return the right answer on small inputs?
> - **Readability** — can a teammate skim it in 30 seconds?
> - **Speed** — does it scale to a million rows without melting?

> **What this does:** Compares a Python `for`-loop sum-of-squares against the
> closed-form formula $\sum_{i=0}^{n-1} i^2 = \frac{(n-1) n (2n-1)}{6}$.
> 
> **Why it matters:** This is the canonical example of *algorithmic*
> speedup beating constant-factor speedup. No amount of vectorisation, no
> GPU, no JIT can beat $O(1)$. Whenever you see a sum, ask "is there a closed form?".
> 
> **What to watch for:** the speedup ratio is enormous (often > 100,000×) but
> it grows with `n`. Algorithmic improvements are *most* valuable on the
> largest inputs — the opposite of micro-optimisations.


### Code cell #7
```python
import time

def slow_sum_squares(n):
    """O(n) loop."""
    s = 0
    for i in range(n):
        s += i * i
    return s

def fast_sum_squares(n):
    """O(1) closed-form: Σ i^2 from 0 to n-1 = (n-1)*n*(2n-1)/6."""
    return (n - 1) * n * (2 * n - 1) // 6

n = 1_000_000
t0 = time.perf_counter(); slow_sum_squares(n); t1 = time.perf_counter()
t2 = time.perf_counter(); fast_sum_squares(n); t3 = time.perf_counter()
print(f"loop:    {t1-t0:.4f}s")
print(f"formula: {t3-t2:.6f}s  -> {int((t1-t0)/(t3-t2))}x speed-up")
```

**What this code does:**

- **Headline:** `import time`
- **Defines:** def `slow_sum_squares` — O(n) loop.; def `fast_sum_squares` — O(1) closed-form: Σ i^2 from 0 to n-1 = (n-1)*n*(2n-1)/6.
- **Key imports:** `import time`

> **Takeaway:** Algorithmic complexity beats micro-optimisation, every time.
> Before you vectorise a loop, ask if the loop is necessary at all.

> ---
> 
> ## 8. Experiments — Experiments

> Try these and observe:
> 
> 1. Replace the `for` loop in `mse` with `sum((a-b)**2 ...)` — does timing change?
> 2. Use a `list` vs a `tuple` for a 1M-element dataset — when does the difference matter?
> 3. Profile a function with `cProfile` — find the hot line.

> **What this does:** Runs our `slow_sum_squares` under `cProfile` and prints
> the top-5 lines by cumulative time. This is the standard way to find the
> *hot line* in any Python script.
> 
> **Why it matters:** Premature optimisation is the root of much evil, but
> *unmeasured* optimisation is worse. Profile before you optimise; optimise
> only what the profile says is hot.
> 
> **What to watch for:** `sort_stats("cumulative")` is what most people want
> for a top-down view; `sort_stats("tottime")` is better when one function is
> called from many places.


### Code cell #8
```python
import cProfile, pstats, io

def profiled():
    return slow_sum_squares(200_000)

pr = cProfile.Profile()
pr.enable()
profiled()
pr.disable()

# Redirect stats to a string so we can print it inline.
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(5)
print(s.getvalue())
```

**What this code does:**

- **Headline:** `import cProfile, pstats, io`
- **Defines:** def `profiled`
- **Key imports:** `import cProfile`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Write a `OneHotEncoder` function that takes a list of categorical labels
>    and returns the one-hot matrix as a list of lists.
> 2. Implement `train_test_split(x, y, test_size=0.2, seed=0)` from scratch.
> 3. Write a `BatchLoader` class that yields mini-batches from a dataset.
> 4. Re-implement `LinearRegressionScratch` to handle multiple features.
> 5. Write a function that detects outliers using the IQR rule.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between a list and a tuple, and when does it matter for ML?
> 2. Explain `*args` and `**kwargs` with an ML-relevant example.
> 3. Why does `is` vs `==` matter when comparing numpy arrays or strings?
> 4. What is the GIL and how does it affect CPU-bound ML code in pure Python?
> 5. How would you memoise an expensive feature-extraction function?
> 6. Describe two ways to iterate over a 1M-row CSV without loading it into memory.
> 7. What does a `defaultdict(list)` buy you when building a label-to-samples map?
> 8. Why is `pathlib.Path` preferable to string concatenation for dataset paths?
