# Contributing to the ML & AI Engineering Cookbook 🧠

Thank you for your interest in contributing to the **ML & AI Engineering Cookbook**! We welcome community contributions, suggestions, corrections, and improvements.

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can You Contribute?](#-how-can-you-contribute)
- [Notebook Pedagogical Structure](#-notebook-pedagogical-structure)
- [Development Workflow](#-development-workflow)
- [Code Style & Standards](#-code-style--standards)
- [Submitting a Pull Request](#-submitting-a-pull-request)

---

## 🤝 Code of Conduct

Be welcoming, respectful, and constructive. We strive to maintain an open and inclusive learning environment for developers and practitioners of all experience levels.

---

## 💡 How Can You Contribute?

You can contribute in several ways:

1. **Enhancing Existing Notebooks**: Add more intuitive diagrams, improve mathematical explanations, or expand the experiments.
2. **Adding Exercises & Interview Questions**: Strengthen the practice challenges and conceptual interview questions in existing modules.
3. **Bug Fixes & Compatibility**: Fix broken links, syntax errors, deprecations, or dependency version conflicts.
4. **New Topics & Case Studies**: Propose or add new hands-on notebooks covering emerging AI architectures, tools, or best practices.
5. **Documentation & Typos**: Improve clarity, fix grammatical errors, or improve module README summaries.

---

## 🧪 Notebook Pedagogical Structure

Each notebook follows a consistent 10-step learning structure wherever applicable:

1. **Concept Overview** – Clear, intuitive explanation of the problem, motivation, and mental model.
2. **Mathematical Foundation** – The core formulas, loss functions, or mathematical derivations.
3. **Implementation from Scratch** – Clean, foundational implementation using pure Python and NumPy.
4. **Library Implementation** – Industry-standard implementation using PyTorch, Scikit-learn, Hugging Face, etc.
5. **Dataset Exploration** – Real-world or reproducible synthetic datasets illustrating the concepts.
6. **Visualization** – Clear plots or diagnostic visualizations explaining the results.
7. **Model Evaluation** – Meaningful metrics (e.g., loss curves, precision/recall, perplexity, latency).
8. **Experiments** – Hyperparameter sweeps or architectural variations showing behavioral trade-offs.
9. **Hands-on Challenges** – Actionable exercises for readers to solve independently.
10. **Interview Questions** – Key theoretical and engineering questions commonly asked in ML/AI interviews.

---

## 🛠️ Development Workflow

### 1. Fork & Clone

```bash
git clone https://github.com/NotHarshhaa/ml-ai-engineering-cookbook.git
cd ml-ai-engineering-cookbook
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run & Test Notebooks

Launch Jupyter locally:
```bash
jupyter notebook
```

Make sure any modified or new notebook executes cleanly without unhandled exceptions. If a notebook requires specialized hardware (e.g., a CUDA GPU) or paid cloud API keys, provide safe mock fallbacks or clearly documented instructions for Google Colab.

---

## 🎨 Code Style & Standards

- **Valid JSON & Clean Metadata**: Ensure Jupyter notebooks (`.ipynb`) have clean execution counts and are valid nbformat 4.
- **Self-Contained**: Use public dataset URLs or library-bundled datasets from the `datasets/` reference.
- **Python Conventions**: Follow PEP 8 guidelines for code clarity.
- **Graceful Fallbacks**: For external services (e.g., LLM APIs, GPU inference), provide offline mock examples or fallback explanations when keys/hardware are unavailable.

---

## 🚀 Submitting a Pull Request

1. Create a descriptive branch:
   ```bash
   git checkout -b feat/module-topic-improvement
   ```
2. Commit your changes with clear commit messages:
   ```bash
   git commit -m "feat(10-generative-ai): improve hybrid search in vector databases notebook"
   ```
3. Push to your fork:
   ```bash
   git push origin feat/module-topic-improvement
   ```
4. Open a Pull Request against the `master` branch with:
   - A summary of what was added or changed.
   - Confirmation that the notebook was tested and executed cleanly.
   - Any screenshots or plots if new visualizations were added.

Thank you for helping make the ML & AI Engineering Cookbook a world-class resource for developers!
