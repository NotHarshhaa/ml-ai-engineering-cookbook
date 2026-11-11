# ML & AI Engineering Cookbook 🧠

**A hands-on collection of Jupyter Notebooks to learn, experiment with, and master Machine Learning and AI Engineering, from fundamentals to production-ready implementations.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/NotHarshhaa/ml-ai-engineering-cookbook)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub Stars](https://img.shields.io/github/stars/NotHarshhaa/ml-ai-engineering-cookbook?style=social)](https://github.com/NotHarshhaa/ml-ai-engineering-cookbook)

---

## 📑 Table of Contents

- [About](#-about)
- [Objectives](#-objectives)
- [Who Is This For?](#-who-is-this-for)
- [Learning Roadmap](#%EF%B8%8F-learning-roadmap)
- [Notebook Structure](#-notebook-learning-structure)
- [Getting Started](#-getting-started)
- [Technologies & Frameworks](#%EF%B8%8F-technologies--frameworks)
- [Repository Structure](#-repository-structure)
- [Learning Philosophy](#-learning-philosophy)
- [Contributing](#-contributing)
- [Useful Resources](#-useful-resources)
- [Project Status](#-project-status)
- [License](#-license)
- [Support](#-support)

## 📖 About

The **ML & AI Engineering Cookbook** is a practical learning repository for building strong foundations and hands-on expertise in Machine Learning, Deep Learning, Generative AI, Agentic AI, MLOps, and AI Infrastructure.

Instead of focusing only on theory, this repository emphasizes implementing algorithms, experimenting with models, understanding the underlying mathematics, working with real-world datasets, and building end-to-end AI applications.

Every topic is explored through a practical `.ipynb` notebook that runs directly in **Google Colab** or locally in Jupyter.

## 🎯 Objectives

- Learn Machine Learning concepts through practical implementations.
- Understand the mathematics behind ML and Deep Learning algorithms.
- Implement algorithms from scratch using Python and NumPy.
- Gain experience with Scikit-learn, PyTorch, and other AI frameworks.
- Explore modern LLM architectures and Generative AI systems.
- Build practical RAG applications and Agentic AI workflows.
- Understand MLOps, LLMOps, model deployment, and monitoring.
- Develop production-oriented AI Engineering skills.

## 👥 Who Is This For?

- Beginners who want a structured path into Machine Learning.
- Software, DevOps, and cloud engineers moving into AI Engineering.
- Practitioners who want runnable references for RAG, agents, MLOps, and LLMOps.
- Anyone preparing for ML and AI engineering interviews.

**Prerequisites:** basic Python knowledge and comfort with high-school-level math. Everything else is introduced progressively.

## 🗺️ Learning Roadmap

The repository is organized into progressive learning modules.

| # | Module | Topics | Status |
|---|---|---|---|
| 01 | [Python for Machine Learning](01-python-for-ml/) | Python, NumPy, Pandas, Statistics, Visualization | ✅ Complete |
| 02 | [ML Fundamentals](02-machine-learning-fundamentals/) | Models, Loss Functions, Optimizers, Gradient Descent | ✅ Complete |
| 03 | [Feature Engineering](03-feature-engineering/) | Data Preprocessing, Feature Selection, Feature Scaling | ✅ Complete |
| 04 | [Supervised Learning](04-supervised-learning/) | Regression, Classification, Decision Trees, Random Forest | ✅ Complete |
| 05 | [Unsupervised Learning](05-unsupervised-learning/) | Clustering, PCA, Dimensionality Reduction | ✅ Complete |
| 06 | [Deep Learning](06-deep-learning/) | Neural Networks, Backpropagation, PyTorch | ✅ Complete |
| 07 | [Computer Vision](07-computer-vision/) | CNNs, Image Classification, Object Detection, YOLO | ✅ Complete |
| 08 | [Natural Language Processing](08-natural-language-processing/) | Tokenization, Embeddings, RNN, LSTM, Attention | ✅ Complete |
| 09 | [Transformers & LLMs](09-transformers-and-llms/) | Transformers, Self-Attention, GPT, BERT, Pretraining | ✅ Complete |
| 10 | [Generative AI Engineering](10-generative-ai/) | Prompt Engineering, RAG, Vector Databases | ✅ Complete |
| 11 | [Agentic AI Engineering](11-agentic-ai/) | AI Agents, Tool Calling, LangGraph, MCP, Multi-Agent Systems | ✅ Complete |
| 12 | [MLOps Engineering](12-mlops/) | MLflow, Experiment Tracking, Model Registry, ML Pipelines | ✅ Complete |
| 13 | [LLMOps & AI Evaluation](13-llmops-and-evaluation/) | LLM Evaluation, Guardrails, Tracing, AI Observability | ✅ Complete |
| 14 | [AI Infrastructure](14-ai-infrastructure/) | GPU Computing, Quantization, vLLM, Model Serving | ✅ Complete |
| 15 | [Fine-Tuning & Advanced AI](15-fine-tuning/) | LoRA, QLoRA, PEFT, SFT, DPO | ✅ Complete |
| 16 | [End-to-End AI Projects](16-end-to-end-projects/) | Real-world ML, GenAI, Agentic AI, and MLOps projects | ✅ Complete |

> Status legend: ✅ Complete · 🚧 In progress · 📝 Planned

## 🧪 Notebook Learning Structure

Each notebook follows a consistent learning approach wherever applicable:

1. **Concept Overview** – Understand the problem and its fundamentals.
2. **Mathematical Foundation** – Explore the underlying mathematics.
3. **Implementation from Scratch** – Build the concept using Python and NumPy.
4. **Library Implementation** – Implement it with industry-standard frameworks.
5. **Dataset Exploration** – Work with practical datasets.
6. **Visualization** – Understand results through plots.
7. **Model Evaluation** – Analyze performance with appropriate metrics.
8. **Experiments** – Modify parameters and observe model behavior.
9. **Hands-on Challenges** – Practice independently.
10. **Interview Questions** – Reinforce conceptual understanding.

## 🚀 Getting Started

### Option 1: Google Colab (no installation)

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Open notebook → GitHub**.
3. Paste the repository path `NotHarshhaa/ml-ai-engineering-cookbook`.
4. Select the notebook you want to practice.
5. Run the cells and start experimenting.

You can also open any notebook directly by replacing `github.com` with `colab.research.google.com/github` in its URL.

### Option 2: Run locally

```bash
# Clone the repository
git clone https://github.com/NotHarshhaa/ml-ai-engineering-cookbook.git
cd ml-ai-engineering-cookbook

# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

> **Note:** Later modules (LLMs, fine-tuning, vLLM) need heavy dependencies and a GPU. Where this applies, the notebook lists its own extra requirements and is best run on Colab.

## 🛠️ Technologies & Frameworks

| Category | Technologies |
|---|---|
| Programming | Python |
| Notebook Environment | Jupyter, Google Colab |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch, TensorFlow |
| NLP | NLTK, spaCy, Hugging Face |
| Generative AI | Hugging Face Transformers, LangChain, LangGraph |
| Vector Databases | FAISS, ChromaDB, Qdrant |
| MLOps | MLflow, DVC |
| LLMOps | LangSmith, Langfuse, Arize Phoenix |
| Model Serving | FastAPI, Ollama, vLLM |

*Tools and frameworks are introduced progressively as the modules evolve.*

## 📂 Repository Structure

```text
ml-ai-engineering-cookbook/
│
├── 01-python-for-ml/
├── 02-machine-learning-fundamentals/
├── 03-feature-engineering/
├── 04-supervised-learning/
├── 05-unsupervised-learning/
├── 06-deep-learning/
├── 07-computer-vision/
├── 08-natural-language-processing/
├── 09-transformers-and-llms/
├── 10-generative-ai/
├── 11-agentic-ai/
├── 12-mlops/
├── 13-llmops-and-evaluation/
├── 14-ai-infrastructure/
├── 15-fine-tuning/
├── 16-end-to-end-projects/
│
├── datasets/
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

Each module contains its own notebooks and supporting resources.

## 💡 Learning Philosophy

**Understand → Implement → Experiment → Evaluate → Build**

The goal is not just to run existing code, but to understand why it works, how it behaves, where it fails, and how it applies to real-world engineering problems.

## 🤝 Contributing

Contributions, suggestions, corrections, and improvements are welcome! You can help by:

- Adding new practical notebooks.
- Improving existing explanations.
- Implementing algorithms from scratch.
- Adding datasets and experiments.
- Fixing bugs and improving notebook execution.
- Suggesting new ML and AI learning topics.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

## 📚 Useful Resources

- [Google Colab](https://colab.research.google.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Hugging Face Papers](https://huggingface.co/papers) and [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent)

## 📌 Project Status

✅ **All modules complete.** All 16 core modules across foundational Machine Learning, Deep Learning, Generative AI, Agentic AI, MLOps, LLMOps, AI Infrastructure, Fine-Tuning, and End-to-End Projects have been fully populated with practical Jupyter notebooks, code implementations, and module guides. Community contributions, experiments, and enhancements are always welcome! See the [roadmap](#%EF%B8%8F-learning-roadmap) for per-module details.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## ⭐ Support

If you find this repository useful, consider giving it a ⭐ on GitHub.

---

**Built with curiosity, experimentation, and a passion for AI Engineering.**

**Maintained by [NotHarshhaa](https://github.com/NotHarshhaa)**
