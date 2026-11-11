# Module 16 — End-to-End AI Projects

The capstone module. Four complete projects that stitch together everything
from modules 02-15 — real ML, GenAI, agentic, and MLOps pipelines you can ship.

| Notebook                                  | Topic                                                                  | Code walkthrough                                                            |
|-------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_end_to_end_ml_churn.ipynb`            | Customer churn: data → features → model → MLflow → FastAPI → drift.  | [README_01_end_to_end_ml_churn.md](README_01_end_to_end_ml_churn.md)        |
| `02_end_to_end_genai_rag.ipynb`           | Document Q&A: chunk → embed → retrieve → prompt → answer → serve.   | [README_02_end_to_end_genai_rag.md](README_02_end_to_end_genai_rag.md)       |
| `03_end_to_end_agentic_research.ipynb`    | Research agent: tools → loop → test → eval → serve → monitor.       | [README_03_end_to_end_agentic_research.md](README_03_end_to_end_agentic_research.md) |
| `04_end_to_end_mlops_pipeline.ipynb`      | Pipeline: load → train → evaluate → register, with caching + guard. | [README_04_end_to_end_mlops_pipeline.md](README_04_end_to_end_mlops_pipeline.md) |

Each notebook has a companion `README_<notebook-name>.md` that walks through
every code cell — open it alongside the notebook for a written explanation.

These four projects are intentionally small enough to read in one sitting but
real enough to be production templates — each can be scaled up to a real
deployable system by swapping the mock components for real libraries (real
LLM, real vector DB, real MLflow).
