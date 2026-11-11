# Code Walkthrough — `03_ml_pipelines.ipynb`

> Module: **12-mlops** | Notebook: **03_ml_pipelines.ipynb**
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

> # 03 — ML Pipelines
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
> ## 1. Concept Overview — ML pipelines

> An ML pipeline is a directed acyclic graph (DAG) of steps: data ingestion →
> preprocessing → training → evaluation → registration. Each step is a
> scripted, versioned, reproducible operation. Pipelines are how production
> ML actually runs — not notebooks.
> 
> **Beginner track:** learn the 5 canonical pipeline steps and how to express
> them as a DAG.
> **Advanced track:** also learn pipeline orchestration tools (Airflow,
> Kubeflow, Prefect, Dagster) and how to make pipelines resumable.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A pipeline is a DAG $G = (V, E)$ where each node $v \in V$ is a function
> $v: \text{inputs} \to \text{outputs}$. Edges $E$ define data flow. Execution
> order is a topological sort of $G$. Each node is *idempotent* — re-running
> it with the same inputs produces the same outputs.
> 
> The pipeline is *resumable* if intermediate outputs are cached; a node
> whose inputs haven't changed is skipped.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a tiny pipeline executor in pure Python:
> nodes are functions, edges are explicit, execution is topological sort.
> Each node's output is cached to disk by input hash for resumability.
> 
> **Why it matters:** every ML pipeline framework (Airflow, Kubeflow, Prefect)
> implements this same pattern. The hard part is distributed execution and
> retry semantics — not the graph.
> 
> **What to watch for:** the input hash determines cache hits. If a function
> depends on a file, hash the file; if it depends on parameters, hash the
> parameters.


### Code cell #1
```python
import hashlib, pickle, os
from pathlib import Path

class MiniPipeline:
    """A toy ML pipeline: nodes are functions, edges define data flow."""
    def __init__(self, cache_dir="pipeline_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.nodes = {}      # name -> (fn, inputs, outputs)
        self.edges = []      # (src, dst)

    def add_node(self, name, fn, inputs=None, outputs=None):
        self.nodes[name] = {"fn": fn, "inputs": inputs or [], "outputs": outputs or []}

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def _hash(self, value):
        return hashlib.md5(pickle.dumps(value)).hexdigest()[:10]

    def run(self, target, **inputs):
        # Topological sort: simple — assume the user added edges in order.
        order = self._topo_sort()
        outputs = dict(inputs)
        for name in order:
            if name not in self.nodes: continue
            node = self.nodes[name]
            in_values = [outputs[v] for v in node["inputs"]]
            # Hash for caching.
            h = self._hash((name, str(in_values), str(inputs)))
            cache_file = self.cache_dir / f"{name}_{h}.pkl"
            if cache_file.exists():
                print(f"  [cache hit] {name}")
                result = pickle.loads(cache_file.read_bytes())
            else:
                print(f"  [executing] {name}")
                if node["inputs"]:
                    result = node["fn"](*in_values)
                else:
                    result = node["fn"](**inputs)
                cache_file.write_bytes(pickle.dumps(result))
            for out in node["outputs"]:
                outputs[out] = result
        return outputs

    def _topo_sort(self):
        # Trivial: order by insertion (works because we add edges in order).
        seen = set()
        order = []
        for src, dst in self.edges:
            if src not in seen: order.append(src); seen.add(src)
            if dst not in seen: order.append(dst); seen.add(dst)
        # Add any nodes without edges.
        for name in self.nodes:
            if name not in seen: order.append(name)
        return order

# Define pipeline steps.
def load_data(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {"X": rng.standard_normal((100, 4)), "y": rng.integers(0, 2, 100)}

def preprocess(data):
    return {"X": (data["X"] - data["X"].mean(0)) / data["X"].std(0), "y": data["y"]}

def train(data):
    # Mock training: just return a "model" as a dict.
    return {"weights": data["X"].mean(0).tolist(), "acc": 0.85}

p = MiniPipeline()
p.add_node("load",      load_data,  inputs=[],        outputs=["raw"])
p.add_node("preprocess", preprocess, inputs=["raw"],  outputs=["clean"])
p.add_node("train",     train,      inputs=["clean"], outputs=["model"])
p.add_edge("load", "preprocess")
p.add_edge("preprocess", "train")

outputs = p.run("model", seed=0)
print("final model:", outputs["model"])
```

**What this code does:**

- **Headline:** `import hashlib, pickle, os`
- **Defines:** class `MiniPipeline` — A toy ML pipeline: nodes are functions, edges define data flow.; def `load_data`; def `preprocess`; def `train`
- **Key imports:** `import hashlib`, `from pathlib import Path`, `import numpy as np`

> **Takeaway:** A pipeline is a DAG of functions with caching. Real
> frameworks add distributed execution, retries, and observability — but
> the core is what you just saw.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `sklearn.pipeline.Pipeline` — the simplest
> production-grade pipeline library. It only handles the *training* part
> (preprocessing + model), not data loading or registration, but it's the
> right starting point.
> 
> **Why it matters:** `sklearn.pipeline.Pipeline` is the most-used pipeline
> library in Python ML. It composes preprocessing + model into a single
> object you can `fit` and `predict` on. It prevents the most common ML bug:
> data leakage between train and test.
> 
> **What to watch for:** the steps run in order; the output of step *i* is
> the input to step *i+1*. The last step must implement `fit` and `predict`;
> earlier steps must implement `fit` and `transform`.


### Code cell #2
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=2)),
    ("clf",    LogisticRegression(max_iter=200)),
])

pipe.fit(X_tr, y_tr)
print(f"pipeline test accuracy: {pipe.score(X_te, y_te):.3f}")
print(f"steps: {[name for name, _ in pipe.steps]}")
```

**What this code does:**

- **Headline:** `from sklearn.pipeline import Pipeline`
- **Key imports:** `from sklearn.pipeline import Pipeline`, `from sklearn.preprocessing import StandardScaler`, `from sklearn.decomposition import PCA`, `from sklearn.linear_model import LogisticRegression`
- **Notable calls:** `LogisticRegression`, `PCA`, `train_test_split`, `load_iris`

> **What this does:** Demonstrates `pipeline.Pipeline` with
> cross-validation + grid search — the standard pattern for hyperparameter
> tuning in production ML.
> 
> **Why it matters:** when you tune hyperparameters, the *pipeline* must be
> cross-validated as a unit. If you tune on `pipe.fit(X_tr)` then evaluate
> on `X_te`, you've leaked — the scaler saw the test set's mean.
> 
> **What to watch for:** `GridSearchCV(pipe, param_grid)` does this
> correctly. The parameter names use `step__param` syntax
> (e.g. `clf__C` for the `C` parameter of the `clf` step).


### Code cell #3
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "pca__n_components": [2, 3, 4],
    "clf__C":            [0.1, 1.0, 10.0],
}
search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
search.fit(X_tr, y_tr)

print(f"best params: {search.best_params_}")
print(f"best CV accuracy: {search.best_score_:.3f}")
print(f"held-out test accuracy: {search.score(X_te, y_te):.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.model_selection import GridSearchCV`
- **Key imports:** `from sklearn.model_selection import GridSearchCV`
- **Notable calls:** `GridSearchCV`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small *churn prediction* dataset (mock) and
> wraps a full pipeline: preprocessing + feature engineering + model + eval.
> This is the kind of pipeline you'd ship to production for a churn model.
> 
> **Why it matters:** churn is the canonical B2B ML use case. The pipeline
> shape (preprocessing + model + eval) is what every tabular ML pipeline
> looks like.
> 
> **What to watch for:** we generate the dataset synthetically so the demo
> is self-contained. In production, the data ingestion step would be a SQL
> query against the data warehouse.


### Code cell #4
```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(0)
n = 1000
churn_df = pd.DataFrame({
    "tenure_months": rng.integers(1, 60, n),
    "monthly_charge": rng.normal(50, 15, n).round(2),
    "support_calls": rng.integers(0, 10, n),
    "contract_type": rng.choice(["monthly", "annual", "biennial"], n),
    "churned": rng.binomial(1, 0.2, n),   # 20% churn rate
})
print(churn_df.head())
print(f"\nchurn rate: {churn_df['churned'].mean():.2%}")
print(f"shape: {churn_df.shape}")
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`, `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises our 3-node pipeline as a DAG.
> 
> **Why it matters:** drawing the pipeline DAG is the standard way to
> communicate it to stakeholders. Production tools (Airflow UI, Prefect
> Cloud) render interactive DAGs.
> 
> **What to watch for:** the arrows show data flow, not control flow.
> Control flow is implicit in the topological order.


### Code cell #5
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

nodes = [
    ("load_data",      1, 1.5, "#3498db"),
    ("preprocess",     4, 1.5, "#9b59b6"),
    ("train",          7, 1.5, "#e74c3c"),
    ("register_model", 10, 1.5, "#27ae60"),
]
for label, x, y, color in nodes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="white", alpha=0.85))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")

for x1, x2 in [(1.8, 3.2), (4.8, 6.2), (7.8, 9.2)]:
    ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

ax.set_title("ML pipeline DAG: load -> preprocess -> train -> register")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import matplotlib.patches as mpatches`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Pipelines are evaluated on:
> 
> - **Per-step outputs**: did each step produce what we expected?
> - **End-to-end performance**: does the final model meet the bar?
> - **Reproducibility**: re-run the pipeline on the same inputs → same output?
> - **Resumability**: if a step fails, can we resume from the last successful step?

> **What this does:** Tests resumability of our MiniPipeline: run it once,
> then run it again and verify the second run uses the cache.
> 
> **Why it matters:** resumability is what makes pipelines *cheap* to
> iterate on. Without it, every code change re-runs the entire pipeline;
> with it, only the changed steps re-run.
> 
> **What to watch for:** if a step's inputs depend on a Python file, hash
> the file content (not the path) — otherwise changing the file doesn't
> invalidate the cache.


### Code cell #6
```python
# Re-run the same pipeline; second run should be all cache hits.
print("=== first run ===")
p.run("model", seed=0)
print("\n=== second run (should be all cache hits) ===")
p.run("model", seed=0)
print("\n=== third run with different seed (preprocess onwards should re-run) ===")
p.run("model", seed=1)
```

**What this code does:**

- **Headline:** `print("=== first run ===")`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add an `evaluate` step that logs metrics to MiniMLflow.
> 2. Make the pipeline fail mid-way (e.g. raise in `train`) and verify the cache lets you resume.
> 3. Express the same pipeline in `prefect` or `kfp` (Kubeflow Pipelines).
> 4. Make the pipeline parameterised: pass `seed` and `n_components` from outside.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement pipeline parallelism: run independent branches concurrently.
> 2. Add retry semantics to each step (with exponential backoff).
> 3. Implement a `DVC`-style data versioning layer so `load_data` is reproducible.
> 4. Build a pipeline that retrains a model weekly on fresh data and auto-registers.
> 5. Implement pipeline observability: log step duration, success/failure, and outputs.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is an ML pipeline, and why is it preferable to a notebook?
> 2. What is a DAG, and why is it the right structure for ML pipelines?
> 3. What is data leakage, and how do pipelines prevent it?
> 4. How do you make a pipeline resumable?
> 5. What is the difference between `sklearn.pipeline.Pipeline` and Airflow?
> 6. How would you test an ML pipeline?
> 7. What is idempotency, and why does it matter for pipelines?
> 8. How would you handle a pipeline step that takes 12 hours to run?
