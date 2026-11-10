# Code Walkthrough — `01_models_overview.ipynb`

> Module: **02-machine-learning-fundamentals** | Notebook: **01_models_overview.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 32
- **Markdown cells:** 27
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

> # 01 — Models and the ML Landscape
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
> ## 1. Concept Overview — What is a model?

> A **model** is a parametric family of functions $f_\theta: \mathcal{X} \to \mathcal{Y}$
> plus a procedure for picking $\theta$ from data. Three pieces define a model:
> 
> 1. **The hypothesis class** — what *shape* of function can the model represent?
> 2. **The loss function** — how do we score a particular $\theta$?
> 3. **The optimisation procedure** — how do we search for the best $\theta$?
> 
> This notebook tours the *hypothesis-class* axis: linear models, kNN, trees,
> kernel methods, neural networks — and the bias/variance trade-off they imply.
> 
> **Beginner track:** focus on the kNN-from-scratch cell and the decision-boundary
> plots — they make the bias/variance story visual.
> **Advanced track:** also reproduce the cross-validation table and add a paired
> t-test on the per-fold scores to test whether model differences are
> statistically significant.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> We frame supervised learning as **Empirical Risk Minimisation**:
> 
> $$\theta^* = \arg\min_\theta \frac{1}{n} \sum_{i=1}^{n} \ell\big(f_\theta(x_i), y_i\big) + \lambda \, \Omega(\theta)$$
> 
> where $\ell$ is the loss and $\Omega$ is a regulariser. The expected risk is
> the expectation over the true data distribution, which we approximate with the
> empirical risk on the training set.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements k-Nearest-Neighbours from scratch with NumPy —
> for each test point, compute the Euclidean distance to every training point,
> take the `k` closest, and majority-vote their labels.
> 
> **Why it matters:** kNN is the simplest non-parametric model and the cleanest
> illustration of *lazy* learning — there is no training step, all the work
> happens at prediction time. Once you can write this, you understand every
> distance-based method (k-means, DBSCAN, kernel SVM).
> 
> **What to watch for:** `np.linalg.norm(X_train - x, axis=1)` is the vectorised
> distance from one test point to every training point — the broadcasting is
> the only thing that makes this fast.


### Code cell #1
```python
# A kNN classifier — the simplest non-parametric model.
import numpy as np

def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        d = np.linalg.norm(X_train - x, axis=1)  # Euclidean dist to every train point
        idx = np.argsort(d)[:k]                  # indices of the k nearest
        votes = y_train[idx]                      # their labels
        labels, counts = np.unique(votes, return_counts=True)  # tally the votes
        preds.append(labels[np.argmax(counts)])  # majority label
    return np.array(preds)

rng = np.random.default_rng(0)               # reproducible synthetic data
X_train = rng.standard_normal((50, 2))
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)  # label: sum > 0?

X_test = np.array([[0.5, 0.5], [-0.5, -0.5]])
print("kNN predictions:", knn_predict(X_train, y_train, X_test, k=5))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `knn_predict`
- **Key imports:** `import numpy as np`

> **Takeaway:** kNN has *no* training — it just stores the data. The hypothesis
> class is implicitly *whatever labels the k nearest neighbours have*, which
> is why kNN is called a *non-parametric* and *lazy* learner.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Trains five scikit-learn classifiers on the `make_moons`
> dataset and reports test accuracy — kNN, logistic regression, a decision
> tree, a random forest, and an RBF-kernel SVM.
> 
> **Why it matters:** This is the *hello world* of model comparison. The five
> models span the bias/variance spectrum: linear (LogReg) → instance-based
> (kNN) → tree → ensemble (RF) → kernel (SVM). Watching their accuracy on the
> same nonlinear dataset teaches you which families fit which shapes.
> 
> **What to watch for:** all models use the same `random_state=0` so the
> comparison is reproducible. `gamma="scale"` is the modern SVM default —
> the old default `gamma="auto"` is almost always wrong.


### Code cell #2
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)  # nonlinear 2-D
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "kNN(5)":         KNeighborsClassifier(5),
    "LogReg":         LogisticRegression(),
    "Tree(d=4)":      DecisionTreeClassifier(max_depth=4, random_state=0),
    "RF(50)":         RandomForestClassifier(n_estimators=50, random_state=0),
    "SVM(rbf)":       SVC(gamma="scale"),
}

for name, model in models.items():
    model.fit(X_tr, y_tr)                                       # train
    acc = accuracy_score(y_te, model.predict(X_te))             # evaluate
    print(f"{name:12s} accuracy = {acc:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.neighbors import KNeighborsClassifier`
- **Key imports:** `from sklearn.neighbors import KNeighborsClassifier`, `from sklearn.linear_model import LogisticRegression`, `from sklearn.tree import DecisionTreeClassifier`, `from sklearn.ensemble import RandomForestClassifier`
- **Notable calls:** `LogisticRegression`, `DecisionTree`, `RandomForest`, `train_test_split`

> **Takeaway:** On `make_moons`, the linear model underfits (bias), the deep
> tree overfits (variance), and the ensemble + kernel methods hit the sweet
> spot. Same data, same metric, *very* different decision boundaries.

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** A single matplotlib scatter of the `make_moons` dataset
> colour-coded by class — the visual we will fit every later model to.
> 
> **Why it matters:** Before fitting anything, *look at the data*. The two
> interleaving crescents explain why a linear classifier must fail and why a
> tree or kernel method succeeds. A 5-second plot saves an hour of confusion.
> 
> **What to watch for:** `edgecolor="white"` is a small styling choice that
> separates overlapping points visually — it makes dense regions readable.


### Code cell #3
```python
# The make_moons dataset — the classic nonlinear boundary benchmark.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="white")  # colour by class
ax.set_title("make_moons dataset"); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the decision boundary of every model trained in
> the previous section side-by-side. For each model we predict the class of
> every point in a 200×200 grid, then colour the regions.
> 
> **Why it matters:** Decision boundaries are the single most informative
> model visualisation. They turn accuracy numbers into shapes — you can
> *see* linear vs piecewise-constant vs smooth-kernel, and *see* overfitting
> as the boundary becoming jagged.
> 
> **What to watch for:** `np.c_[xx.ravel(), yy.ravel()]` is the standard idiom
> for turning a meshgrid into a list of points to predict on; the result is
> reshaped back to the grid with `.reshape(xx.shape)`.


### Code cell #4
```python
# Decision boundaries of all five models side by side.
fig, axes = plt.subplots(1, len(models), figsize=(20, 4))

xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]   # 40000 x 2 list of (x, y) points to predict on

for ax, (name, model) in zip(axes, models.items()):
    zz = model.predict(grid).reshape(xx.shape)         # predict on every grid point
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")  # fill the decision regions
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15, edgecolor="white")
    ax.set_title(name)
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `fig, axes = plt.subplots(1, len(models), figsize=(20, 4))`

> **Takeaway:** The linear model draws a straight line; the tree draws
> axis-aligned rectangles; the random forest smooths the tree; the SVM draws
> a curved maximum-margin boundary. Same data, same metric, *different
> inductive biases* — and that is the whole story of model selection.

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> All five models were evaluated on accuracy. But accuracy hides a lot:
> 
> - On imbalanced data, accuracy rewards majority-class predictors.
> - Two models with the same accuracy can have very different boundaries.
> - We need cross-validation to know if a difference is real.

> **What this does:** Runs 5-fold cross-validation on every model and reports
> the mean and standard deviation of accuracy across folds. This tells us
> whether an accuracy difference is real or just sampling noise.
> 
> **Why it matters:** A single train/test split can rank models in any order
> by luck. Cross-validation averages over 5 different splits and reports
> the spread — the standard deviation is what tells you if two models'
> means are meaningfully different.
> 
> **What to watch for:** `cross_val_score` *clones* the model internally, so
> it is fine to pass in an already-fitted model — your fit will be discarded.
> Always pass `scoring="accuracy"` (or your metric of choice) explicitly.


### Code cell #5
```python
from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")  # 5-fold CV
    print(f"{name:12s} CV acc = {scores.mean():.3f} +/- {scores.std():.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.model_selection import cross_val_score`
- **Key imports:** `from sklearn.model_selection import cross_val_score`
- **Notable calls:** `cross_val_score`

> **Takeaway:** If two models' CV means differ by less than 1 standard
> deviation, treat them as tied. A difference that is small relative to the
> fold-to-fold spread is almost certainly noise, not skill.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Increase the noise in `make_moons` from 0.25 to 0.45 — which model degrades fastest?
> 2. Repeat with 30 samples instead of 300 — what does the bias/variance picture look like?
> 3. Add 5 noisy features — which model is most robust?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a `MajorityClassifier` that always predicts the most common class.
> 2. Plot the bias/variance trade-off by varying tree depth from 1 to 15.
> 3. Build a function `compare_models(models, X, y, cv=5)` that returns a tidy DataFrame.
> 4. Add a regularisation path: vary `C` in logistic regression and plot train/test accuracy.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Define *parametric* vs *non-parametric* models with one example each.
> 2. What is the bias/variance trade-off, and how does regularisation affect it?
> 3. Why is kNN called a *lazy* learner?
> 4. When does a decision tree outperform a logistic regression? When does it not?
> 5. What is the no-free-lunch theorem, and what does it imply for model selection?
> 6. Explain Empirical Risk Minimisation in one sentence.
> 7. What is the difference between inductive bias and the bias in bias/variance?
> 8. Why is cross-validation preferable to a single train/test split?
