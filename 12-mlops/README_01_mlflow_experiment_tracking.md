# Code Walkthrough — `01_mlflow_experiment_tracking.ipynb`

> Module: **12-mlops** | Notebook: **01_mlflow_experiment_tracking.ipynb**
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

> # 01 — MLflow Experiment Tracking
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
> ## 1. Concept Overview — Experiment tracking with MLflow

> Every model you train should record its hyperparameters, metrics, code
> version, and artifacts. MLflow is the de-facto standard for this — a few
> lines of code, a local file-based UI, and you have a reproducible experiment
> log.
> 
> **Beginner track:** learn the `start_run / log_param / log_metric /
> log_artifact` quartet. That's 80% of MLflow.
> **Advanced track:** also learn MLflow Model Registry, autologging, and
> model serving with MLflow Models.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> An MLflow *experiment* contains *runs*. Each run is a tuple
> $(P, M, A, \text{meta})$ where $P$ is a set of parameters, $M$ is a set of
> metrics, $A$ is a set of artifacts (files), and $\text{meta}$ includes
> timestamps, git commit, and Python env.
> 
> Comparison: query runs by metric, sort, find the best. This is the *minimum*
> infrastructure to make ML engineering reproducible.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a tiny experiment tracker in pure Python —
> a `MiniMLflow` class that stores runs as JSON. Same shape as the real
> MLflow API: `start_run`, `log_param`, `log_metric`, `log_artifact`.
> 
> **Why it matters:** MLflow's API is conceptually trivial. The value is in
> the *UI* and *storage backend*, not the API. Once you've reimplemented it,
> the real MLflow stops feeling magical.
> 
> **What to watch for:** we store everything in a dict-of-dicts. Real MLflow
> uses a relational DB (SQLite, PostgreSQL) for queryability.


### Code cell #1
```python
import json, time, os
from pathlib import Path

class MiniMLflow:
    """A toy MLflow: stores runs in a JSON file."""
    def __init__(self, store_path="minimlf.json"):
        self.path = Path(store_path)
        if self.path.exists():
            self.store = json.loads(self.path.read_text())
        else:
            self.store = {"experiments": {}}
        self.current_run = None

    def set_experiment(self, name):
        self.store["experiments"].setdefault(name, {"runs": {}})
        self.exp_name = name

    def start_run(self, run_name=None):
        run_id = f"run_{int(time.time()*1000)}"
        self.current_run = run_id
        self.store["experiments"][self.exp_name]["runs"][run_id] = {
            "name": run_name or run_id,
            "params": {}, "metrics": {}, "artifacts": [],
            "start_time": time.time(),
        }
        return run_id

    def log_param(self, k, v):
        self.store["experiments"][self.exp_name]["runs"][self.current_run]["params"][k] = v

    def log_metric(self, k, v):
        self.store["experiments"][self.exp_name]["runs"][self.current_run]["metrics"][k] = v

    def log_artifact(self, path):
        self.store["experiments"][self.exp_name]["runs"][self.current_run]["artifacts"].append(path)

    def end_run(self):
        run = self.store["experiments"][self.exp_name]["runs"][self.current_run]
        run["end_time"] = time.time()
        self.path.write_text(json.dumps(self.store, indent=2))
        self.current_run = None

    def search_runs(self, order_by=None):
        runs = self.store["experiments"][self.exp_name]["runs"]
        items = list(runs.values())
        if order_by:
            # order_by = "metrics.accuracy DESC" style
            key = order_by.split()[0].split(".")[1]
            items.sort(key=lambda r: -r["metrics"].get(key, 0))
        return items

# Demo: log 3 runs with different hyperparameters.
mlf = MiniMLflow()
mlf.set_experiment("iris_logreg")
for C in [0.1, 1.0, 10.0]:
    mlf.start_run(f"C={C}")
    mlf.log_param("C", C)
    mlf.log_param("solver", "liblinear")
    # Mock accuracy that depends on C.
    acc = 0.85 + 0.05 * (1 if C == 1.0 else 0) - 0.02 * (1 if C > 5 else 0)
    mlf.log_metric("accuracy", acc)
    mlf.log_metric("loss", 1 - acc + 0.1)
    mlf.end_run()

best = mlf.search_runs(order_by="metrics.accuracy DESC")[0]
print("best run:", best["name"], " accuracy:", best["metrics"]["accuracy"])
```

**What this code does:**

- **Headline:** `import json, time, os`
- **Defines:** class `MiniMLflow` — A toy MLflow: stores runs in a JSON file.
- **Key imports:** `import json`, `from pathlib import Path`

> **Takeaway:** MLflow is just structured JSON logging. The hard part is
> actually *using* it consistently across every experiment — not the API.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses the real `mlflow` library to log a scikit-learn
> experiment. We enable `autolog` which captures hyperparameters, metrics,
> and the model automatically. If `mlflow` isn't installed, we skip the
> demo.
> 
> **Why it matters:** `autolog` is the *one-line* version of experiment
> tracking — call `mlflow.sklearn.autolog()` before `fit()` and MLflow logs
> everything. It's the lowest-effort, highest-value ML tool you'll ever
> use.
> 
> **What to watch for:** autolog captures the *fitted model* as an artifact,
> so you can later load it with `mlflow.sklearn.load_model(run_id)`. That's
> the foundation of model reproducibility.


### Code cell #2
```python
try:
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    mlflow.set_experiment("iris_logreg")
    mlflow.sklearn.autolog()                  # capture everything automatically

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    with mlflow.start_run(run_name="autolog_demo"):
        clf = LogisticRegression(C=1.0, max_iter=200, solver="liblinear").fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        mlflow.log_metric("test_accuracy", acc)
    print(f"logged run with test_accuracy = {acc:.3f}")
    print("view UI with: mlflow ui  (then open http://localhost:5000)")
except ImportError as e:
    print(f"mlflow not installed ({e}) — pip install mlflow")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import mlflow`, `import mlflow.sklearn`, `from sklearn.linear_model import LogisticRegression`, `from sklearn.datasets import load_iris`
- **Notable calls:** `LogisticRegression`, `train_test_split`, `load_iris`

> **What this does:** Manually logs parameters and metrics (without
> autolog). This is what you'd use when you need fine-grained control —
> e.g. logging custom metrics like training latency or model size.
> 
> **Why it matters:** autolog is convenient but opinionated. For custom
> metrics (inference latency, model size, fairness metrics), you need the
> manual API.
> 
> **What to watch for:** always log *both* `params` (input) and `metrics`
> (output). Parameters are immutable; metrics can be a time series (use
> `log_metric(key, value, step=step)` for curves).


### Code cell #3
```python
try:
    from sklearn.ensemble import RandomForestClassifier
    import time

    with mlflow.start_run(run_name="manual_log_demo"):
        for n_est in [10, 50, 100]:
            for max_depth in [3, 5, None]:
                # Use nested runs to log each hyperparameter combination.
                with mlflow.start_run(nested=True):
                    mlflow.log_param("n_estimators", n_est)
                    mlflow.log_param("max_depth", max_depth if max_depth else -1)
                    t0 = time.perf_counter()
                    clf = RandomForestClassifier(n_estimators=n_est,
                                                 max_depth=max_depth,
                                                 random_state=0).fit(X_tr, y_tr)
                    train_time = time.perf_counter() - t0
                    acc = clf.score(X_te, y_te)
                    mlflow.log_metric("test_accuracy", acc)
                    mlflow.log_metric("train_seconds", train_time)
    print("logged nested runs; view with `mlflow ui`")
except NameError:
    print("mlflow not available — see previous cell")
except Exception as e:
    print(f"skipped nested-runs demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from sklearn.ensemble import RandomForestClassifier`, `import time`
- **Notable calls:** `RandomForest`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads a small regression dataset (California Housing
> subset) for the MLOps demos. We'll train several models on it and log
> them all to MLflow.
> 
> **Why it matters:** MLOps is fundamentally about *many* runs of the same
> task. A small, fast dataset lets you run 50 experiments in seconds — the
> kind of iteration MLOps enables.
> 
> **What to watch for:** we cap to 1000 rows for speed. Real MLOps uses
> full datasets — but the patterns are the same.


### Code cell #4
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X_full, y_full = data.data, data.target
X = X_full.iloc[:1000]; y = y_full.iloc[:1000]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"train: {X_tr.shape}, test: {X_te.shape}")
print(f"target range: {y.min():.2f} to {y.max():.2f}")
```

**What this code does:**

- **Headline:** `from sklearn.datasets import fetch_california_housing`
- **Key imports:** `from sklearn.datasets import fetch_california_housing`, `from sklearn.model_selection import train_test_split`
- **Notable calls:** `train_test_split`, `fetch_california_housing`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the Pareto front of our MiniMLflow runs: test
> accuracy vs training time. The Pareto front shows which runs are
> non-dominated — neither faster nor more accurate than another.
> 
> **Why it matters:** the Pareto front is the *honest* way to compare
> models with multiple objectives. A model is on the front if no other
> model is both faster and more accurate.
> 
> **What to watch for:** real MLflow UIs plot this interactively. The static
> version is what you'd put in a report.


### Code cell #5
```python
import matplotlib.pyplot as plt
import numpy as np

# Generate mock runs with accuracy + latency trade-off.
rng = np.random.default_rng(0)
n_runs = 30
accs = rng.uniform(0.80, 0.95, n_runs)
times = rng.uniform(0.05, 2.0, n_runs)

# Compute Pareto front: a run is dominated if another has both better acc and lower time.
def pareto_mask(accs, times):
    keep = np.ones(len(accs), dtype=bool)
    for i in range(len(accs)):
        for j in range(len(accs)):
            if i == j: continue
            if accs[j] >= accs[i] and times[j] <= times[i] and (accs[j] > accs[i] or times[j] < times[i]):
                keep[i] = False
                break
    return keep

pareto = pareto_mask(accs, times)
plt.figure(figsize=(7, 5))
plt.scatter(times[~pareto], accs[~pareto], color="lightgray", label="dominated")
plt.scatter(times[pareto], accs[pareto], color="steelblue", s=80, label="Pareto front", edgecolor="white")
plt.xlabel("training time (s)"); plt.ylabel("test accuracy")
plt.title("Pareto front of model runs")
plt.legend(); plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Defines:** def `pareto_mask`
- **Key imports:** `import matplotlib.pyplot as plt`, `import numpy as np`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> MLflow doesn't evaluate models — it logs the evaluations. The metrics you
> log become the basis for model selection. Common patterns:
> 
> - **Single metric**: pick the run with the best `test_accuracy`.
> - **Pareto**: pick runs on the accuracy/latency Pareto front.
> - **Constrained**: pick the run with `latency < 100ms` that has best accuracy.

> **What this does:** Queries our MiniMLflow for the best run by accuracy,
> and prints its params + metrics. This is the *minimum* query you'd run
> before deploying a model.
> 
> **Why it matters:** without this query, you can't reliably pick a model
> for deployment. The whole point of experiment tracking is to make this
> query fast and trustworthy.
> 
> **What to watch for:** we sort by accuracy descending and take the first
> run. In a real MLOps pipeline you'd also filter by training time, model
> size, and fairness metrics.


### Code cell #6
```python
# Find the best run from our MiniMLflow.
runs = mlf.search_runs(order_by="metrics.accuracy DESC")
print(f"total runs: {len(runs)}")
print(f"best run: {runs[0]['name']}, accuracy: {runs[0]['metrics']['accuracy']:.3f}")
print(f"params: {runs[0]['params']}")
print(f"all run accuracies: {[r['metrics']['accuracy'] for r in runs]}")
```

**What this code does:**

- **Headline:** `runs = mlf.search_runs(order_by="metrics.accuracy DESC")`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Log a custom artifact: save a confusion-matrix PNG with `mlflow.log_artifact`.
> 2. Compare 4 models (LR, RF, GBM, SVM) in the same experiment.
> 3. Use `mlflow.sklearn.autolog(log_models=True)` and reload the model from a run ID.
> 4. Set up MLflow tracking server (`mlflow server`) and log to a remote URI.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a custom MLflow `flavor` for a domain-specific model type.
> 2. Build a CI job that fails if a new run does not beat the current champion.
> 3. Implement a learning-curve logger that logs `loss(step)` as a time series.
> 4. Set up MLflow Model Registry and promote a model to production stage.
> 5. Implement experiment comparison across different git commits.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What problem does MLflow solve?
> 2. What is the difference between `params`, `metrics`, and `artifacts`?
> 3. What is MLflow autologging, and when would you not use it?
> 4. How do you compare runs across different experiments?
> 5. What is the MLflow Model Registry, and when do you need it?
> 6. How would you back MLflow with a remote database?
> 7. What is a Pareto front, and why is it useful for model selection?
> 8. How do you reproduce a model from a year-old MLflow run?
