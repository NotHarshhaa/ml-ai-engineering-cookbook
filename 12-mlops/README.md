# Module 12 — MLOps Engineering

The infrastructure that turns notebooks into production ML systems. Experiment
tracking, model registry, pipelines, and the promotion guards that keep
deployments safe.

| Notebook                                  | Topic                                                                  | Code walkthrough                                                            |
|-------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_mlflow_experiment_tracking.ipynb`     | Mini-MLflow + real MLflow, autolog, params/metrics/artifacts, Pareto. | [README_01_mlflow_experiment_tracking.md](README_01_mlflow_experiment_tracking.md) |
| `02_model_registry.ipynb`                 | Registry state machine, promotion guards, lineage.                  | [README_02_model_registry.md](README_02_model_registry.md)                    |
| `03_ml_pipelines.ipynb`                   | Pipeline DAG, caching, resumability, sklearn.Pipeline + GridSearch. | [README_03_ml_pipelines.md](README_03_ml_pipelines.md)                        |

Each notebook has a companion `README_<notebook-name>.md` that walks through
every code cell — open it alongside the notebook for a written explanation.
