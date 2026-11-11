# Code Walkthrough — `04_end_to_end_mlops_pipeline.ipynb`

> Module: **16-end-to-end-projects** | Notebook: **04_end_to_end_mlops_pipeline.ipynb**
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

> # 04 — End-to-End MLOps Pipeline
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
> ## 1. Concept Overview — End-to-end MLOps: notebook to production pipeline

> This notebook converts the churn notebook into a *production pipeline*:
> automated training, evaluation, registration, and deployment. Each step
> is a function; the pipeline runs them in order with caching and
> observability.
> 
> **Beginner track:** follow the 4 steps: define pipeline steps → wire them
> into a DAG → add caching → run end-to-end.
> **Advanced track:** also add CI/CD triggers, alerting, and rollback.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> An MLOps pipeline is a DAG of steps, each producing artifacts consumed by
> the next. Reproducibility comes from caching by (step_name, input_hash).
> Reliability comes from idempotency — running a step twice with the same
> inputs produces the same outputs.

> ---
> 
> ## 3. Implementation from Scratch — Step 1 — Define pipeline steps

> **What this does:** Defines 4 pipeline steps as pure functions:
> `load_data`, `train_model`, `evaluate_model`, `register_model`. Each takes
> inputs and produces a typed output. Pure functions are easy to test and
> cache.
> 
> **Why it matters:** each step should be a *pure function* — no side effects,
> no global state. This makes them cacheable, testable, and runnable in
> isolation. Real ML pipelines are built this way.
> 
> **What to watch for:** in production, `load_data` reads from a data
> warehouse; `register_model` calls MLflow. The function signatures stay
> the same.


### Code cell #1
```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def load_data(seed=42):
    """Step 1: load and clean the data. Returns (X, y)."""
    rng = np.random.default_rng(seed)
    n = 2000
    tenure = rng.integers(1, 60, n)
    monthly = rng.normal(50, 15, n).clip(20, 200)
    support = rng.integers(0, 10, n)
    contract = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])
    logit = -2 + 0.02 * (60 - tenure) + 0.03 * (monthly - 50) + 0.4 * support - 1.0 * contract
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)
    X = pd.DataFrame({"tenure": tenure, "monthly": monthly, "support": support, "contract": contract})
    return X, y

def train_model(X, y, n_estimators=100):
    """Step 2: train a model. Returns a fitted pipeline."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe

def evaluate_model(model, X_te, y_te):
    """Step 3: evaluate. Returns a dict of metrics."""
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc":  float(average_precision_score(y_te, proba)),
    }

def register_model(model, metrics, model_name="churn_v1"):
    """Step 4: register the model. Returns the registry entry."""
    # In production: mlflow.sklearn.log_model + create_model_version.
    return {"name": model_name, "metrics": metrics, "status": "registered"}

# Smoke-test each step.
X, y = load_data()
print(f"step 1 ok: X={X.shape}, y={y.shape}")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
m = train_model(X_tr, y_tr)
print(f"step 2 ok: model trained")
mets = evaluate_model(m, X_te, y_te)
print(f"step 3 ok: {mets}")
reg = register_model(m, mets)
print(f"step 4 ok: {reg}")
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Defines:** def `load_data` — Step 1: load and clean the data. Returns (X, y).; def `train_model` — Step 2: train a model. Returns a fitted pipeline.; def `evaluate_model` — Step 3: evaluate. Returns a dict of metrics.; def `register_model` — Step 4: register the model. Returns the registry entry.
- **Key imports:** `import pandas as pd`, `import numpy as np`, `from sklearn.pipeline import Pipeline`, `from sklearn.preprocessing import StandardScaler`
- **Notable calls:** `train_test_split`

> **Takeaway:** MLOps pipeline = pure functions + DAG + caching. Each step is
> small and testable. The complexity is in *stitching* them, not in the
> steps themselves.

> ---
> 
> ## 4. Library Implementation — Step 2 — Build the pipeline with caching

> **What this does:** Wires the 4 steps into a pipeline with input-hash
> caching. Re-running the pipeline with the same inputs hits the cache for
> unchanged steps; only changed steps re-run.
> 
> **Why it matters:** caching is what makes pipelines *cheap* to iterate
> on. Without it, every code change re-runs the entire pipeline; with it,
> only the changed step and its downstream re-run.
> 
> **What to watch for:** `joblib.Memory` or `diskcache` provide production-
> grade caching. Our implementation is a simple dict-based version.


### Code cell #2
```python
import hashlib, pickle
from pathlib import Path

CACHE_DIR = Path("pipeline_cache"); CACHE_DIR.mkdir(exist_ok=True)

def hash_args(args):
    return hashlib.md5(pickle.dumps(args)).hexdigest()[:10]

def cached_step(name, fn, args):
    h = hash_args((name, args))
    cache_file = CACHE_DIR / f"{name}_{h}.pkl"
    if cache_file.exists():
        print(f"  [cache hit] {name}")
        return pickle.loads(cache_file.read_bytes())
    print(f"  [executing] {name}")
    result = fn(**args)
    cache_file.write_bytes(pickle.dumps(result))
    return result

def run_pipeline(seed=42, n_estimators=100):
    """Run the full pipeline with caching."""
    print("--- pipeline run ---")
    X, y = cached_step("load_data", load_data, {"seed": seed})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = cached_step("train_model", train_model,
                       {"X": X_tr, "y": y_tr, "n_estimators": n_estimators})
    metrics = cached_step("evaluate_model", evaluate_model,
                          {"model": model, "X_te": X_te, "y_te": y_te})
    reg = register_model(model, metrics)
    return reg

# Run twice; second run should hit cache for unchanged steps.
print("=== first run ===")
print(run_pipeline())
print("\n=== second run (same args; cache hits) ===")
print(run_pipeline())
```

**What this code does:**

- **Headline:** `import hashlib, pickle`
- **Defines:** def `hash_args`; def `cached_step`; def `run_pipeline` — Run the full pipeline with caching.
- **Key imports:** `import hashlib`, `from pathlib import Path`
- **Notable calls:** `train_test_split`

> ---
> 
> ## 5. Dataset Exploration — Step 3 — Promote to production with a guard

> **What this does:** Adds a promotion guard: only register the model if
> its ROC-AUC beats the current production model's. This prevents
> regressions from reaching production.
> 
> **Why it matters:** promotion guards are how you make MLOps *safe*. A
> pipeline that always registers the latest model will ship regressions.
> 
> **What to watch for:** real guards have multiple criteria — accuracy,
> latency, fairness, calibration. We check only ROC-AUC for the demo.


### Code cell #3
```python
class ModelRegistry:
    """Toy registry with a current-production model."""
    def __init__(self):
        self.production = None
    def set_production(self, entry):
        self.production = entry
    def maybe_promote(self, candidate, threshold=0.0):
        if self.production is None:
            self.set_production(candidate)
            return True, "auto-promoted (first model)"
        candidate_auc = candidate["metrics"]["roc_auc"]
        current_auc = self.production["metrics"]["roc_auc"]
        if candidate_auc > current_auc + threshold:
            self.set_production(candidate)
            return True, f"promoted ({candidate_auc:.3f} > {current_auc:.3f})"
        return False, f"rejected ({candidate_auc:.3f} <= {current_auc:.3f})"

reg = ModelRegistry()
# Mock a "current production" model with ROC-AUC = 0.85.
reg.set_production({"name": "churn_v0", "metrics": {"roc_auc": 0.85}})
# Candidate 1: ROC-AUC = 0.88 -> promoted.
c1 = {"name": "churn_v1", "metrics": {"roc_auc": 0.88}}
print(f"promote v1? {reg.maybe_promote(c1)}")
# Candidate 2: ROC-AUC = 0.82 -> rejected.
c2 = {"name": "churn_v2", "metrics": {"roc_auc": 0.82}}
print(f"promote v2? {reg.maybe_promote(c2)}")
print(f"current production: {reg.production}")
```

**What this code does:**

- **Headline:** `class ModelRegistry:`
- **Defines:** class `ModelRegistry` — Toy registry with a current-production model.

> **Takeaway:** A promotion guard is the simplest MLOps safety. Without it,
> you'll eventually ship a regression. With it, every promotion is a
> deliberate, audited decision.

> ---
> 
> ## 6. Visualization — Step 4 — End-to-end with guard

> **What this does:** Wires the pipeline + registry + guard into one
> end-to-end script. Run it twice with different hyperparameters; the guard
> decides whether each candidate reaches production.
> 
> **Why it matters:** this is the *minimum* production ML pipeline. With a
> scheduler (Airflow, Prefect) running it weekly, you have a fully automated
> ML system.
> 
> **What to watch for:** in production, this script would be a CI job,
> triggered by a git push or a schedule. The guard's decision would be
> logged and alert a human for review.


### Code cell #4
```python
registry = ModelRegistry()
# Mock the current production with a baseline AUC.
registry.set_production({"name": "churn_baseline", "metrics": {"roc_auc": 0.85}})

# Run 3 candidates with different hyperparameters.
for n_est in [50, 100, 150]:
    print(f"\n=== candidate with n_estimators={n_est} ===")
    X, y = cached_step("load_data", load_data, {"seed": 42})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_tr, y_tr, n_estimators=n_est)
    metrics = evaluate_model(model, X_te, y_te)
    candidate = {"name": f"churn_n{n_est}", "metrics": metrics}
    print(f"  metrics: {metrics}")
    promoted, reason = registry.maybe_promote(candidate)
    print(f"  -> {reason}")

print(f"\nfinal production model: {registry.production}")
```

**What this code does:**

- **Headline:** `registry = ModelRegistry()`
- **Notable calls:** `train_test_split`

> ---
> 
> ## 7. Model Evaluation — Step 5 — Observability and alerting

> **What this does:** Logs every pipeline run to a metrics file. In
> production, this would be a Prometheus push or a Langfuse event.
> 
> **Why it matters:** without observability, you can't tell when the
> pipeline degrades. ROC-AUC dropping from 0.90 to 0.85 over 4 weeks is a
> slow drift that no one notices without metrics.
> 
> **What to watch for:** real observability tracks not just final metrics
> but intermediate ones — data freshness, data quality, model latency,
> prediction distribution.


### Code cell #5
```python
import json, time
from pathlib import Path

metrics_log_file = Path("pipeline_metrics.jsonl")

def log_pipeline_run(metrics, promoted):
    entry = {"timestamp": time.time(), "metrics": metrics, "promoted": promoted}
    with open(metrics_log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

# Log a few runs.
for auc in [0.88, 0.82, 0.91]:
    m = {"roc_auc": auc, "pr_auc": auc - 0.05}
    log_pipeline_run(m, promoted=(auc > 0.85))

# Read back and report.
print("pipeline metrics history:")
for line in metrics_log_file.read_text().strip().split("\n"):
    entry = json.loads(line)
    print(f"  roc_auc={entry['metrics']['roc_auc']:.3f}  promoted={entry['promoted']}")
```

**What this code does:**

- **Headline:** `import json, time`
- **Defines:** def `log_pipeline_run`
- **Key imports:** `import json`, `from pathlib import Path`

> ---
> 
> ## 8. Experiments — Step 6 — CI integration

> **What this does:** The CI pattern: this script would run on every git push
> to the main branch. If the guard rejects, the build is green (no model
> was promoted). If it accepts, the build is also green (a new model was
> promoted). Failures (data drift, training error) make the build red.
> 
> **Why it matters:** CI is how you make MLOps *continuous*. Without CI,
> MLOps is manual ops.
> 
> **What to watch for:** real CI runs in GitHub Actions, GitLab CI, or
> Jenkins. The script doesn't change — only the runner does.


### Code cell #6
```python
# A CI script (run on git push):
#   python run_pipeline.py
# Then check:
#   1. exit code 0 (no errors)
#   2. metrics within expected range (data sanity check)
#   3. promotion decision logged

# Mock CI run.
print("=== CI pipeline run ===")
try:
    result = run_pipeline(seed=42, n_estimators=100)
    print(f"CI: pipeline succeeded. metrics: {result.get('metrics')}")
    print("CI: build green (regardless of promotion decision)")
except Exception as e:
    print(f"CI: pipeline FAILED: {e}")
    print("CI: build red (human investigation needed)")
```

**What this code does:**

- **Headline:** `print("=== CI pipeline run ===")`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Replace the toy pipeline with `prefect` or `kfp` (Kubeflow Pipelines).
> 2. Add a `data_drift_check` step that fails the pipeline if drift is detected.
> 3. Set up GitHub Actions to run the pipeline on every push.
> 4. Add Slack alerts on promotion / failure.
> 5. Implement rollback: revert to the previous production model on failure.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Walk through the components of a production ML pipeline.
> 2. Why use pure functions for pipeline steps?
> 3. What is a promotion guard, and why do you need one?
> 4. How does pipeline caching save time?
> 5. What is the role of CI in MLOps?
> 6. How would you test an ML pipeline end-to-end?
> 7. How would you set up alerting for a pipeline?
> 8. What is the difference between Airflow, Prefect, and Kubeflow?
