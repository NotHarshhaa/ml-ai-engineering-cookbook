# Code Walkthrough — `02_logistic_regression.ipynb`

> Module: **04-supervised-learning** | Notebook: **02_logistic_regression.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 23
- **Markdown cells:** 18
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

> # 02 — Logistic Regression
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
> ## 1. Concept Overview — Logistic regression — the probabilistic classifier

> Logistic regression models $P(y = 1 \mid x) = \sigma(w^\top x + b)$ where
> $\sigma(z) = 1 / (1 + e^{-z})$. Despite the name, it is a *classification*
> model — and the building block of every neural-network classifier.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Model**: $p(x) = \sigma(w^\top x + b)$
> 
> **Loss (cross-entropy / negative log-likelihood)**:
> $$L(w) = -\sum_i y_i \log p(x_i) + (1-y_i) \log (1 - p(x_i))$$
> 
> **Gradient**: $\nabla_w L = \sum_i (p(x_i) - y_i) \, x_i$ — a beautifully
> simple form: the gradient is the prediction error times the input.
> 
> **Multinomial** (softmax): $p_c(x) = \frac{e^{w_c^\top x}}{\sum_k e^{w_k^\top x}}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

def sigmoid(z):
    # numerically stable
    return np.where(z >= 0, 1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=500, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
    def fit(self, X, y):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            p = sigmoid(Xb @ self.w)
            grad = Xb.T @ (p - y) / len(y) + self.l2 * self.w
            self.w -= self.lr * grad
        return self
    def predict_proba(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return sigmoid(Xb @ self.w)
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 2))
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

lr = LogisticRegressionScratch(lr=0.5, epochs=1000, l2=0.01).fit(X, y)
print("weights:", lr.w.round(3))
print("train accuracy:", (lr.predict(X) == y).mean())
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `sigmoid`; class `LogisticRegressionScratch`
- **Key imports:** `import numpy as np`
- **Notable calls:** `LogisticRegression`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for name, model in [
    ("logreg (no-reg)",  LogisticRegression(penalty=None, max_iter=5000)),
    ("logreg L2",        LogisticRegression(penalty="l2", C=1.0, max_iter=5000)),
    ("logreg L1",        LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=5000)),
]:
    pipe = Pipeline([("sc", StandardScaler()), ("clf", model)]).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    print(f"{name:18s} acc={accuracy_score(y_te, p>=0.5):.3f} auc={roc_auc_score(y_te, p):.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.linear_model import LogisticRegression`
- **Key imports:** `from sklearn.linear_model import LogisticRegression`, `from sklearn.datasets import load_breast_cancer`, `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import accuracy_score`
- **Notable calls:** `LogisticRegression`, `cross_val_score`, `train_test_split`, `load_breast_cancer`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
import pandas as pd
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y
print("shape:", df.shape)
print("class balance:", df.target.value_counts(normalize=True).round(3).to_dict())
print("\ntop 5 |correlations| with target:")
print(df.corr()["target"].abs().sort_values(ascending=False).head(5))
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

pipe = Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=5000))]).fit(X_tr, y_tr)
p = pipe.predict_proba(X_te)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fpr, tpr, _ = roc_curve(y_te, p)
axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_te, p):.3f}")
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curve"); axes[0].legend()

prec, rec, _ = precision_recall_curve(y_te, p)
axes[1].plot(rec, prec)
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("Precision-Recall curve")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `from sklearn.metrics import roc_curve`
- **Notable calls:** `LogisticRegression`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For binary classifiers, accuracy is not enough. Use:
> 
> - **ROC-AUC** — ranking quality, threshold-independent.
> - **Precision, recall, F1** — for imbalanced classes.
> - **Confusion matrix** — to see *which* errors happen.
> - **Calibration** — does $p = 0.8$ really mean 80%?


### Code cell #5
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = (p >= 0.5).astype(int)
print(classification_report(y_te, y_pred))

cm = confusion_matrix(y_te, y_pred)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot()
plt.title("Confusion matrix at threshold 0.5")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay`
- **Key imports:** `from sklearn.metrics import confusion_matrix`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Increase the threshold to 0.7 — how do precision and recall move?
> 2. Add class weights `class_weight="balanced"` — does it help on an artificially imbalanced version?
> 3. Train with L1 and check how many coefficients end up exactly zero.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **softmax regression** from scratch for the 3-class iris problem.
> 2. Implement **L2-regularised logistic regression** using Newton's method.
> 3. Add an `is_calibrated` plot — binned predicted probabilities vs observed frequencies.
> 4. Implement **F1-score threshold optimisation** on the validation set.
> 5. Implement **SGD-based logistic regression** for a 1M-row dataset.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why is logistic regression called *regression* if it is a classifier?
> 2. Why is cross-entropy the natural loss for a Bernoulli output?
> 3. What is the gradient of the log-loss with respect to $w$?
> 4. When would L1-regularised logistic regression beat L2?
> 5. How does the threshold affect precision/recall trade-off?
> 6. What is a calibrated classifier, and how do you calibrate one?
> 7. What is multicollinearity's effect on logistic-regression coefficients?
> 8. How do you handle class imbalance in logistic regression?
