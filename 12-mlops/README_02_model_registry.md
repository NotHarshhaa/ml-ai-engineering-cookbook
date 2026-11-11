# Code Walkthrough — `02_model_registry.ipynb`

> Module: **12-mlops** | Notebook: **02_model_registry.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 29
- **Markdown cells:** 24
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

> # 02 — Model Registry
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
> ## 1. Concept Overview — Model registry

> A model registry is a versioned, role-controlled store for trained models.
> It's where the *best* model from your experiment tracking graduates to a
> candidate for deployment. MLflow Model Registry is the standard in Python;
> WEKA, Vertex AI, SageMaker have their own equivalents.
> 
> **Beginner track:** learn the model lifecycle: `None → Staging →
> Production → Archived`. That's the registry in one line.
> **Advanced track:** also learn model lineage, model signing, and webhooks
> for stage transitions.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A model registry is a partially ordered set of *model versions* with
> metadata: `(name, version, stage, source_uri, tags)`. Stages are a state
> machine: `None → Staging → Production → Archived`. Transitions can be
> guarded by approvals or CI checks.
> 
> Lineage: a model version points to the run that produced it; the run
> points to the dataset and code version. This forms the *reproducibility
> graph*.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a tiny model registry: `register`,
> `transition_stage`, `get_latest`. Models are stored as dict entries.
> 
> **Why it matters:** a registry is just a database table with a state
> machine. The complexity in real registries is access control and audit
> logging — not the data model.
> 
> **What to watch for:** we don't enforce *one production model per name*.
> Real registries do; if you transition a new model to Production, the old
> one auto-archives.


### Code cell #1
```python
class MiniRegistry:
    """A toy model registry: dict-of-dicts with a stage state machine."""
    def __init__(self):
        self.models = {}  # name -> list of versions
    def register(self, name, source_uri, tags=None):
        version = len(self.models.get(name, [])) + 1
        self.models.setdefault(name, []).append({
            "version": version,
            "stage": "None",
            "source_uri": source_uri,
            "tags": tags or {},
        })
        return version
    def transition_stage(self, name, version, new_stage):
        # In a real registry, transitioning to Production auto-archives the
        # previous production version.
        if new_stage == "Production":
            for v in self.models[name]:
                if v["stage"] == "Production":
                    v["stage"] = "Archived"
        for v in self.models[name]:
            if v["version"] == version:
                v["stage"] = new_stage
                return v
    def get_latest(self, name, stage="Production"):
        for v in reversed(self.models.get(name, [])):
            if v["stage"] == stage:
                return v
        return None

reg = MiniRegistry()
v1 = reg.register("iris_clf", "runs:/abc/model", tags={"framework": "sklearn"})
v2 = reg.register("iris_clf", "runs:/def/model", tags={"framework": "sklearn"})
reg.transition_stage("iris_clf", v1, "Staging")
reg.transition_stage("iris_clf", v2, "Production")

print(f"v1 stage: {reg.models['iris_clf'][0]['stage']}")  # Staging
print(f"v2 stage: {reg.models['iris_clf'][1]['stage']}")  # Production
print(f"latest production: {reg.get_latest('iris_clf')}")
```

**What this code does:**

- **Headline:** `class MiniRegistry:`
- **Defines:** class `MiniRegistry` — A toy model registry: dict-of-dicts with a stage state machine.

> **Takeaway:** A registry is just a state machine on top of a model store.
> The hard part is the workflow (approvals, hooks, lineage), not the data
> model.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses the real `mlflow.tracking.MlflowClient` to
> register a model, transition it through stages, and load it back. If
> mlflow isn't installed, we skip.
> 
> **Why it matters:** the `MlflowClient` API is the production-grade way to
> manage models. Knowing it lets you script the full promotion pipeline.
> 
> **What to watch for:** `create_registered_model` creates the name once;
> `create_model_version` adds a new version under that name. Calling
> `create_registered_model` twice with the same name raises an error.


### Code cell #2
```python
try:
    from mlflow.tracking import MlflowClient
    import mlflow

    client = MlflowClient()
    name = "iris_demo_registry"
    # Try to create; if it exists, ignore.
    try:
        client.create_registered_model(name)
    except Exception:
        pass

    # Register the latest run's model under this name.
    runs = client.search_runs(experiment_ids=[mlflow.get_experiment_by_name("iris_logreg").experiment_id])
    if runs:
        run_id = runs[0].info.run_id
        mv = client.create_model_version(name=name, source=f"runs:/{run_id}/model",
                                          run_id=run_id)
        client.transition_model_version_stage(name, mv.version, "Staging")
        print(f"registered {name} v{mv.version}, stage=Staging")
    else:
        print("no runs to register; run the previous notebook first")
except Exception as e:
    print(f"skipped MLflow registry demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from mlflow.tracking import MlflowClient`, `import mlflow`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Simulates a typical model lifecycle: 5 trainings of
> the same model, with the registry promoting the best one to production.
> 
> **Why it matters:** this is the *cadence* of real MLOps — train, evaluate,
> promote (or archive). Doing it 5 times in a notebook makes the loop
> obvious.
> 
> **What to watch for:** we promote based on test accuracy. Real registries
> have *much* more complex promotion criteria — fairness, latency,
> robustness, business metrics.


### Code cell #3
```python
# Simulate 5 trainings and register each; promote the best to Production.
import numpy as np
rng = np.random.default_rng(0)

reg2 = MiniRegistry()
for i in range(5):
    # Mock accuracy.
    acc = 0.85 + rng.uniform(-0.05, 0.05)
    version = reg2.register("iris_clf", f"runs:/run_{i}/model",
                            tags={"test_acc": round(acc, 3)})
    print(f"  v{version} registered with test_acc={acc:.3f}")

# Promote the best (highest test_acc).
best_v = max(reg2.models["iris_clf"], key=lambda v: v["tags"]["test_acc"])
reg2.transition_stage("iris_clf", best_v["version"], "Production")
print(f"\npromoted v{best_v['version']} (test_acc={best_v['tags']['test_acc']:.3f}) to Production")
print(f"current production: {reg2.get_latest('iris_clf')}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the model lifecycle state machine:
> None → Staging → Production → Archived, with allowed transitions.
> 
> **Why it matters:** the state machine is the *contract* of the registry.
> Every team that uses the registry needs to agree on which transitions are
> allowed and who approves them.
> 
> **What to watch for:** some workflows allow "rollback" (Production →
> Staging). Others force a new version. Both are valid choices.


### Code cell #4
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

states = [
    ("None",        1, 1.5, "#95a5a6"),
    ("Staging",     4, 1.5, "#f39c12"),
    ("Production",  7, 1.5, "#27ae60"),
    ("Archived",   10, 1.5, "#7f8c8d"),
]
for label, x, y, color in states:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="white", alpha=0.85))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

arrows = [(1.8, 1.5, 3.2, 1.5), (4.8, 1.5, 6.2, 1.5), (7.8, 1.5, 9.2, 1.5)]
for x1, y1, x2, y2 in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.set_title("Model registry state machine")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import matplotlib.patches as mpatches`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> A registry doesn't evaluate models — it tracks their *promotion*.
> Evaluation happens *before* each transition:
> 
> - **None → Staging**: model passed automated tests (CI).
> - **Staging → Production**: model beat the current production model on
>   the eval set + business metrics.
> - **Production → Archived**: replaced by a better model, or aged out.

> **What this does:** A simple promotion guard: only transition to
> Production if the candidate's test_acc beats the current production's
> test_acc.
> 
> **Why it matters:** promotion guards are how you avoid regressions in
> production. Without them, a bad model can replace a good one and nobody
> notices until the business metrics drop.
> 
> **What to watch for:** real guards have *multiple* criteria — accuracy,
> latency, fairness, calibration. One metric is rarely enough.


### Code cell #5
```python
def promotion_guard(registry, name, candidate_version, metric="test_acc"):
    current = registry.get_latest(name, stage="Production")
    candidate = None
    for v in registry.models[name]:
        if v["version"] == candidate_version:
            candidate = v
            break
    if candidate is None:
        return False, "candidate not found"
    if current is None:
        return True, "no current production model — auto-promote"
    if candidate["tags"][metric] > current["tags"][metric]:
        return True, f"candidate {candidate['tags'][metric]:.3f} > current {current['tags'][metric]:.3f}"
    return False, f"candidate {candidate['tags'][metric]:.3f} <= current {current['tags'][metric]:.3f}"

# Try to promote v1 (which we know is not the best).
ok, reason = promotion_guard(reg2, "iris_clf", 1)
print(f"promote v1? {ok}  reason: {reason}")
ok, reason = promotion_guard(reg2, "iris_clf", best_v["version"])
print(f"promote v{best_v['version']} (the best)? {ok}  reason: {reason}")
```

**What this code does:**

- **Headline:** `def promotion_guard(registry, name, candidate_version, metric="test_acc"):`
- **Defines:** def `promotion_guard`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a `latency_ms` tag and require `<100ms` for production promotion.
> 2. Implement a *rollback* function that returns to the previous production model.
> 3. Use MLflow webhooks to trigger CI on a stage transition.
> 4. Add model signing (`mlflow.models.sign_model`) to verify model integrity.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a multi-criteria promotion guard (accuracy + latency + fairness).
> 2. Build a CI pipeline that automatically registers a model on every git push.
> 3. Implement a model-rollback API that restores the previous production model.
> 4. Add an approval step: a human must approve Staging → Production.
> 5. Implement model lineage tracking (model → run → dataset → git commit).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between experiment tracking and model registry?
> 2. What are the standard model lifecycle stages?
> 3. How do you prevent a regression when promoting a model?
> 4. What is model lineage, and why does it matter?
> 5. How would you implement a rollback?
> 6. What is model signing, and what threat does it mitigate?
> 7. How do you handle multiple teams sharing one registry?
> 8. What is the difference between MLflow Model Registry and Vertex AI Model Registry?
