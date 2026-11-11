# Code Walkthrough — `01_end_to_end_ml_churn.ipynb`

> Module: **16-end-to-end-projects** | Notebook: **01_end_to_end_ml_churn.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 32
- **Markdown cells:** 25
- **Code cells:** 7

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

> # 01 — End-to-End ML: Customer Churn
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
> ## 1. Concept Overview — End-to-end ML project: customer churn prediction

> This notebook brings together everything from modules 02-12: data loading,
> feature engineering, model training, evaluation, MLOps, and deployment.
> We'll build a real customer-churn predictor end-to-end.
> 
> **Beginner track:** follow the 6 steps from data to deployed model. Each
> step is small enough to read in one sitting.
> **Advanced track:** also do the experiments — try LightGBM, add SHAP for
> explainability, and deploy with FastAPI.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Customer churn is binary classification: $P(\text{churn} \mid x)$, where $x$
> includes tenure, monthly charge, support calls, contract type. Loss: binary
> cross-entropy. Metric: ROC-AUC + recall at low FPR (catching churners is
> worth more than false alarms).
> 
> Production constraint: must produce a probability per customer per day, in
> <50 ms, on 1M customers.

> ---
> 
> ## 3. Implementation from Scratch — Step 1 — Data ingestion and cleaning

> **What this does:** Loads a synthetic churn dataset (the same one from
> module 12), explores it, and handles missing values. This is the *first*
> step of any ML project.
> 
> **Why it matters:** data quality determines model quality. A model can't
> learn from a column that's 50% missing or has the wrong dtype.
> 
> **What to watch for:** in production, this step reads from a data warehouse
> (Snowflake, BigQuery). Here we generate synthetic data so the demo is
> self-contained.


### Code cell #1
```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n_customers = 5000

tenure = rng.integers(1, 60, n_customers)
monthly_charge = rng.normal(50, 15, n_customers).clip(20, 200)
support_calls = rng.integers(0, 10, n_customers)
contract_type = rng.choice(["monthly", "annual", "biennial"], n_customers, p=[0.5, 0.3, 0.2])

# Churn probability depends on features (so the model has signal to learn).
churn_logit = (
    -2.0
    + 0.02 * (60 - tenure)         # newer customers churn more
    + 0.03 * (monthly_charge - 50) # higher charges -> more churn
    + 0.4 * support_calls          # support issues drive churn
    - 1.0 * (contract_type != "monthly")  # contracts reduce churn
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churned = (rng.uniform(size=n_customers) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": np.arange(1, n_customers+1),
    "tenure_months": tenure,
    "monthly_charge": monthly_charge.round(2),
    "support_calls": support_calls,
    "contract_type": contract_type,
    "churned": churned,
})
# Inject 5% missing values in monthly_charge (to demo cleaning).
miss_idx = rng.choice(n_customers, size=int(0.05 * n_customers), replace=False)
df.loc[miss_idx, "monthly_charge"] = np.nan

print(f"shape: {df.shape}")
print(f"churn rate: {df['churned'].mean():.2%}")
print(f"missing per column:\n{df.isna().sum()}")
df.head()
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`, `import numpy as np`

> **What this does:** Cleans the data: imputes missing `monthly_charge` with
> the median per `contract_type` group, encodes `contract_type` ordinally
> (month < annual < biennial).
> 
> **Why it matters:** group-wise imputation is more accurate than global
> imputation — annual customers have different charge distributions than
> monthly ones.
> 
> **What to watch for:** `contract_type` is *ordinal* here — there's a
> natural order (month → annual → biennial). Don't one-hot encode an ordinal
> feature.


### Code cell #2
```python
# Group-wise imputation.
df["monthly_charge"] = df.groupby("contract_type")["monthly_charge"].transform(
    lambda s: s.fillna(s.median())
)
# Ordinal-encode contract_type (monthly=0, annual=1, biennial=2).
df["contract_code"] = df["contract_type"].map({"monthly": 0, "annual": 1, "biennial": 2})

# Final clean dataframe.
clean = df.drop(columns=["customer_id", "contract_type"]).copy()
print(f"clean shape: {clean.shape}, remaining nulls: {clean.isna().sum().sum()}")
clean.head()
```

**What this code does:**

- **Headline:** `df["monthly_charge"] = df.groupby("contract_type")["monthly_charge"].transform(`

> **Takeaway:** Data cleaning = imputation + encoding + sanity checks.
> Without it, no model can rescue the project.

> ---
> 
> ## 4. Library Implementation — Step 2 — Feature engineering & preprocessing pipeline

> **What this does:** Builds a scikit-learn pipeline that:
> 1. Standardises numeric features.
> 2. Trains a LogisticRegression + RandomForest + GradientBoosting.
> 3. Reports cross-validated ROC-AUC for each.
> 
> **Why it matters:** the pipeline pattern (preprocessing + model in one
> object) is what prevents data leakage and makes the model deployable.
> 
> **What to watch for:** we use `Pipeline` + `ColumnTransformer` — the
> production-grade pattern. The whole pipeline is `fit` and `predict`-able
> as a single object.


### Code cell #3
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

X = clean.drop(columns="churned")
y = clean["churned"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

numeric_features = ["tenure_months", "monthly_charge", "support_calls", "contract_code"]
pre = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
])

models = {
    "logreg":   LogisticRegression(max_iter=1000),
    "rf":       RandomForestClassifier(n_estimators=100, random_state=0),
    "gbm":      GradientBoostingClassifier(n_estimators=100, random_state=0),
}
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
    print(f"  {name:8s} CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.pipeline import Pipeline`
- **Key imports:** `from sklearn.pipeline import Pipeline`, `from sklearn.compose import ColumnTransformer`, `from sklearn.preprocessing import StandardScaler`, `from sklearn.linear_model import LogisticRegression`
- **Notable calls:** `LogisticRegression`, `RandomForest`, `cross_val_score`, `train_test_split`

> ---
> 
> ## 5. Dataset Exploration — Step 3 — Final model training and evaluation

> **What this does:** Trains the best model (GBM, based on CV) on the full
> training set and evaluates on the held-out test set with ROC-AUC, PR-AUC,
> and a confusion matrix.
> 
> **Why it matters:** the test set is the *only* honest measure of model
> quality. CV tells you which model is best; the test set tells you how good
> the best one actually is.
> 
> **What to watch for:** for churn, ROC-AUC > 0.85 is good; > 0.90 is
> excellent. Our mock data has strong signal, so we should hit ~0.90.


### Code cell #4
```python
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

best_pipe = Pipeline([("pre", pre), ("model", GradientBoostingClassifier(n_estimators=100, random_state=0))])
best_pipe.fit(X_tr, y_tr)

y_proba = best_pipe.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, y_proba)
pr_auc = average_precision_score(y_te, y_proba)
print(f"test ROC-AUC: {auc:.3f}")
print(f"test PR-AUC : {pr_auc:.3f}")
print(classification_report(y_te, best_pipe.predict(X_te)))
```

**What this code does:**

- **Headline:** `from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, c...`
- **Key imports:** `from sklearn.metrics import roc_auc_score`

> ---
> 
> ## 6. Visualization — Step 4 — MLOps: log to MLflow and register

> **What this does:** Logs the trained model and its metrics to MLflow. In
> production, this step runs in CI on every git push — the model registry
> tracks versions and stages.
> 
> **Why it matters:** without MLflow (or equivalent), you can't reproduce a
> model from 6 months ago, can't tell which version is in production, can't
> rollback a bad model. MLOps is the foundation of reproducible ML.
> 
> **What to watch for:** `log_metrics` accepts a dict; `log_params` accepts
> any JSON-serialisable value. `log_model` saves the entire pipeline so you
> can reload it with `mlflow.sklearn.load_model`.


### Code cell #5
```python
try:
    import mlflow
    import mlflow.sklearn
    mlflow.set_experiment("churn_predictor")
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run(run_name="gbm_baseline"):
        mlflow.log_params({"n_estimators": 100, "model": "GradientBoostingClassifier"})
        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.sklearn.log_model(best_pipe, "model")
    print(f"logged run with ROC-AUC = {auc:.3f}")
    print("(view at: mlflow ui  -> http://localhost:5000)")
except ImportError as e:
    print(f"mlflow not installed ({e}) — skipping MLOps demo")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import mlflow`, `import mlflow.sklearn`

> ---
> 
> ## 7. Model Evaluation — Step 5 — Serving: deploy with FastAPI

> **What this does:** Builds a FastAPI service that wraps the trained
> pipeline: `POST /predict` accepts customer features and returns a churn
> probability.
> 
> **Why it matters:** a model in a notebook is a prototype. A served model
> is what production looks like. This is the final step of every ML project.
> 
> **What to watch for:** `best_pipe` is loaded as a module-level variable so
> it's loaded once at server startup, not per request. In production, save
> the model with MLflow and load it on startup.


### Code cell #6
```python
from pydantic import BaseModel, Field

class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., ge=1, le=120)
    monthly_charge: float = Field(..., ge=0, le=1000)
    support_calls: int = Field(..., ge=0, le=50)
    contract_code: int = Field(..., ge=0, le=2)

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool

# The FastAPI app would be:
# app = FastAPI()
# @app.post("/predict", response_model=ChurnResponse)
# def predict(req: ChurnRequest):
#     X = pd.DataFrame([req.dict()])
#     prob = best_pipe.predict_proba(X)[0, 1]
#     return ChurnResponse(churn_probability=float(prob),
#                         churn_prediction=bool(prob > 0.5))

# Mock-call the predict function.
sample = ChurnRequest(tenure_months=24, monthly_charge=55.0, support_calls=2, contract_code=0)
X_sample = pd.DataFrame([sample.model_dump()])
prob = float(best_pipe.predict_proba(X_sample)[0, 1])
print(f"sample: {sample}")
print(f"churn probability: {prob:.3f}  prediction: {prob > 0.5}")
```

**What this code does:**

- **Headline:** `from pydantic import BaseModel, Field`
- **Defines:** class `ChurnRequest`; class `ChurnResponse`
- **Key imports:** `from pydantic import BaseModel`

> ---
> 
> ## 8. Experiments — Step 6 — Monitoring & drift detection

> **What this does:** Implements a simple drift detector: compare the
> feature distribution of recent requests to the training distribution.
> If they differ significantly, alert.
> 
> **Why it matters:** models silently degrade when the input distribution
> shifts (e.g. economic downturn changes customer behaviour). Without drift
> detection, you discover this only when business metrics drop.
> 
> **What to watch for:** we use the Kolmogorov-Smirnov test (KS) for numeric
> features. A p-value < 0.05 means the distributions differ significantly.


### Code cell #7
```python
from scipy.stats import ks_2samp

# Mock production traffic: shifted distribution.
rng = np.random.default_rng(1)
production_traffic = pd.DataFrame({
    "tenure_months": rng.integers(1, 30, 200),       # newer customers
    "monthly_charge": rng.normal(70, 20, 200).clip(20, 200),  # higher charges
    "support_calls": rng.integers(0, 15, 200),        # more support calls
    "contract_code": rng.choice([0, 1, 2], 200, p=[0.7, 0.2, 0.1]),  # more monthly
})

# KS test per feature: compare training vs production distribution.
for col in numeric_features:
    stat, p = ks_2samp(X_tr[col], production_traffic[col])
    drift = "DRIFT!" if p < 0.05 else "ok"
    print(f"  {col:20s}  KS stat={stat:.3f}  p={p:.4f}  {drift}")
```

**What this code does:**

- **Headline:** `from scipy.stats import ks_2samp`
- **Key imports:** `from scipy.stats import ks_2samp`

> **Takeaway:** End-to-end ML = data → features → model → MLOps → serving →
> monitoring. Each step is small; the complexity is in *stitching* them.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Replace GradientBoostingClassifier with XGBoost or LightGBM — does AUC improve?
> 2. Add SHAP values for per-customer explainability (why did the model predict churn?).
> 3. Deploy the FastAPI app with Docker + uvicorn.
> 4. Set up a retraining pipeline: retrain weekly on fresh data, promote via MLflow registry.
> 5. Add a `version` endpoint that returns the model version + last-trained timestamp.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Walk through the steps of an end-to-end ML project.
> 2. How would you handle a customer-churn model that degrades over time?
> 3. What is the role of MLOps in a production ML system?
> 4. How do you choose between logistic regression, RF, GBM, and XGBoost for churn?
> 5. What metrics would you track for a churn predictor in production?
> 6. How would you explain a churn prediction to a non-technical stakeholder?
> 7. What is drift detection, and when do you retrain?
> 8. How would you deploy a churn model to handle 1M predictions/day?
