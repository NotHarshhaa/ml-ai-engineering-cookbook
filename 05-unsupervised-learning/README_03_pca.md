# Code Walkthrough — `03_pca.ipynb`

> Module: **05-unsupervised-learning** | Notebook: **03_pca.ipynb**
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

> # 03 — Principal Component Analysis
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
> ## 1. Concept Overview — PCA — Principal Component Analysis

> PCA finds orthogonal directions of maximum variance in centred data and
> projects onto the first $k$ of them. It is the classic dimensionality
> reduction — used for visualisation, denoising, feature compression, and as
> a regulariser downstream of high-dimensional features.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Let $X \in \mathbb{R}^{n \times d}$ be centred (column means = 0). The
> covariance matrix is $\Sigma = \frac{1}{n-1} X^\top X$.
> 
> Eigendecomposition: $\Sigma = V \Lambda V^\top$ where $\Lambda = \text{diag}(\lambda_1 \ge \lambda_2 \ge \ldots)$.
> 
> The $k$-dimensional projection is $Z = X V_k$ where $V_k$ is the first $k$
> columns of $V$.
> 
> Explained variance ratio: $\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

def pca_scratch(X, k=2):
    # 1. centre
    mu = X.mean(axis=0)
    Xc = X - mu
    # 2. covariance (n-1 normalization)
    C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    # 3. eigendecomposition (sorted desc)
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V = eigvecs[:, order]
    # 4. project
    Z = Xc @ V[:, :k]
    return Z, V[:, :k], eigvals

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 5)) @ np.array([[3,1,0,0,0],
                                                 [1,2,0,0,0],
                                                 [0,0,1,0,0],
                                                 [0,0,0,0.5,0],
                                                 [0,0,0,0,0.1]])
Z, V, eigvals = pca_scratch(X, k=2)
print("eigenvalues (sorted):", eigvals.round(3))
print("explained-variance ratio:", (eigvals / eigvals.sum()).round(3))
print("projection shape:", Z.shape)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `pca_scratch`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)  # 8x8 = 64 features
pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)

print("explained variance ratio (top 2):", pca.explained_variance_ratio_.round(3))

plt.figure(figsize=(7, 6))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
plt.colorbar(scatter, label="digit class")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA of MNIST 8x8 digits")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.decomposition import PCA`
- **Key imports:** `from sklearn.decomposition import PCA`, `from sklearn.datasets import load_digits`, `import matplotlib.pyplot as plt`
- **Notable calls:** `PCA`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)
X = iris.drop(columns="species").values
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=4).fit(Xs)

print("explained variance ratio per PC:", pca.explained_variance_ratio_.round(3))
print("cumulative:", np.cumsum(pca.explained_variance_ratio_).round(3))
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`, `from sklearn.preprocessing import StandardScaler`
- **Notable calls:** `PCA`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# scree plot
axes[0].bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, label="individual")
axes[0].plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), "ro-", label="cumulative")
axes[0].set_xlabel("principal component"); axes[0].set_ylabel("variance ratio")
axes[0].set_title("Scree plot"); axes[0].legend()

# 2-D projection
Z = pca.transform(Xs)
for sp in iris["species"].unique():
    m = iris["species"] == sp
    axes[1].scatter(Z[m, 0], Z[m, 1], label=sp, edgecolor="white")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title("PCA 2-D projection"); axes[1].legend()
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `fig, axes = plt.subplots(1, 2, figsize=(13, 5))`
- **Notable calls:** `PCA`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> We evaluate PCA by:
> 
> - **Cumulative explained variance** — how many components to reach 95%?
> - **Reconstruction error** — $\|X - X V_k V_k^\top\|^2$ decreases with $k$.
> - **Downstream model performance** — does PCA hurt or help kNN?


### Code cell #5
```python
# PCA reconstruction error vs k.
errs = []
for k in range(1, 5):
    Vk = PCA(n_components=k).fit(Xs).components_[:k]
    recon = Xs @ Vk.T @ Vk
    errs.append(np.mean((Xs - recon) ** 2))
plt.plot(range(1, 5), errs, "o-")
plt.xlabel("k"); plt.ylabel("reconstruction MSE")
plt.title("PCA reconstruction error")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `errs = []`
- **Notable calls:** `PCA`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Skip standardisation (use raw `X` instead of `Xs`) — how do the components change?
> 2. Use `svd_solver="arpack"` (truncated SVD) on a 50k x 5000 matrix.
> 3. Train a kNN on the first 2 vs first 4 PCs of iris — compare accuracy.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **incremental PCA** that processes batches of 1000 rows.
> 2. Implement **kernel PCA** with an RBF kernel and use it on `make_moons`.
> 3. Implement **probabilistic PCA** with the EM algorithm.
> 4. Use **sparse PCA** (`sklearn.decomposition.SparsePCA`) and contrast
>    interpretability against vanilla PCA.
> 5. Implement PCA via SVD instead of eigendecomposition and explain why SVD
>    is numerically preferable.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why do we centre the data before PCA? What happens if we do not?
> 2. Why is standardisation often necessary before PCA?
> 3. What is the relationship between PCA and SVD?
> 4. What does an explained-variance ratio of 0.95 in the first 2 PCs mean?
> 5. Why is PCA not good for non-linear manifolds, and what is the alternative?
> 6. How does PCA help with the curse of dimensionality?
> 7. What is the difference between PCA and Factor Analysis?
> 8. How would you choose the number of components for a downstream classifier?
