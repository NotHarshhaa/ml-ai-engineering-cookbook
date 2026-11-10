# Code Walkthrough — `04_dimensionality_reduction.ipynb`

> Module: **05-unsupervised-learning** | Notebook: **04_dimensionality_reduction.ipynb**
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

> # 04 — Other Dimensionality Reduction
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
> ## 1. Concept Overview — Beyond PCA — non-linear dimensionality reduction

> PCA is linear: it can only undo rotations and rescalings. Real data often
> lies on a non-linear manifold (a swiss-roll, a curve, a sphere). The methods
> in this notebook recover such structure: kernel PCA, t-SNE, UMAP, MDS,
> Isomap, LLE.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **t-SNE** — converts pairwise similarities to probabilities and matches
> high-D and low-D distributions via KL divergence.
> 
> **UMAP** — assumes data is uniformly distributed on a Riemannian manifold;
> uses fuzzy simplicial sets and cross-entropy.
> 
> **MDS** — preserves pairwise distances in low-D (classical MDS = PCA on the
> Gram matrix).
> 
> **Isomap** — geodesic distances via kNN graph + MDS.
> 
> **LLE** — each point is a linear combination of its neighbours; preserve the
> same weights in low-D.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
# Classical MDS from scratch — equivalent to PCA on the Gram matrix.
import numpy as np

def classical_mds(D, k=2):
    # double-centre the squared distance matrix to get a Gram matrix
    D2 = D ** 2
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1][:k]
    L = np.diag(np.sqrt(np.maximum(eigvals[order], 0)))
    return eigvecs[:, order] @ L

rng = np.random.default_rng(0)
X = rng.standard_normal((30, 5))
D = np.linalg.norm(X[:, None] - X, axis=2)
Z = classical_mds(D, k=2)
print("MDS embedding shape:", Z.shape)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `classical_mds`
- **Key imports:** `import numpy as np`
- **Notable calls:** `PCA`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

X, color = make_swiss_roll(n_samples=1500, noise=0.1, random_state=0)

methods = {
    "PCA(2)":       PCA(n_components=2).fit_transform(X),
    "KernelPCA":   KernelPCA(n_components=2, kernel="rbf", gamma=0.05).fit_transform(X),
    "MDS":         MDS(n_components=2, random_state=0).fit_transform(X),
    "Isomap":      Isomap(n_components=2, n_neighbors=10).fit_transform(X),
    "LLE":         LocallyLinearEmbedding(n_components=2, n_neighbors=10).fit_transform(X),
    "t-SNE":       TSNE(n_components=2, random_state=0, perplexity=30,
                         init="pca", learning_rate="auto").fit_transform(X),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (name, Z) in zip(axes.flat, methods.items()):
    ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap="Spectral", s=8)
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.decomposition import PCA, KernelPCA`
- **Key imports:** `from sklearn.decomposition import PCA`, `from sklearn.manifold import TSNE`, `from sklearn.datasets import make_swiss_roll`, `import matplotlib.pyplot as plt`
- **Notable calls:** `PCA`, `TSNE`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
from sklearn.datasets import fetch_openml
# MNIST 70000 x 784.  For speed we use 3000 samples.
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
idx = np.random.default_rng(0).choice(len(mnist.data), 3000, replace=False)
X = mnist.data[idx] / 255.0
y = mnist.target[idx].astype(int)
print("subset shape:", X.shape)
```

**What this code does:**

- **Headline:** `from sklearn.datasets import fetch_openml`
- **Key imports:** `from sklearn.datasets import fetch_openml`
- **Notable calls:** `fetch_openml`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
from sklearn.manifold import TSNE
import umap  # pip install umap-learn — fall back to t-SNE if missing
try:
    Z = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
    name = "UMAP"
except Exception:
    Z = TSNE(n_components=2, random_state=0, init="pca",
             learning_rate="auto", perplexity=30).fit_transform(X)
    name = "t-SNE (umap-learn not installed)"

plt.figure(figsize=(8, 7))
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=4, alpha=0.7)
plt.colorbar(label="digit")
plt.title(f"{name} of MNIST subset")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.manifold import TSNE`
- **Key imports:** `from sklearn.manifold import TSNE`, `import umap  # pip install umap-learn — fall back to t-SNE if missing`
- **Notable calls:** `TSNE`, `UMAP`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Non-linear DR methods are mostly *qualitative* — the goal is a sensible 2-D
> embedding for visualisation. Quantitative checks:
> 
> - **Trustworthiness** (`sklearn.manifold.trustworthiness`) — does kNN
>   neighbourhood structure survive the projection?
> - **Downstream classification accuracy** on the embedding.
> - **Stability across random seeds** — t-SNE is famously seed-sensitive.


### Code cell #5
```python
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Trustworthiness of the 2-D UMAP / t-SNE embedding.
tw = trustworthiness(X, Z, n_neighbors=5)
print(f"trustworthiness @ k=5: {tw:.3f}")

# kNN classification accuracy on the 2-D embedding vs 64-D PCA.
pca2 = PCA(n_components=2).fit_transform(X)
acc_emb = cross_val_score(KNeighborsClassifier(5), Z, y, cv=3).mean()
acc_pca = cross_val_score(KNeighborsClassifier(5), pca2, y, cv=3).mean()
print(f"kNN acc on {name}: {acc_emb:.3f}")
print(f"kNN acc on PCA-2D: {acc_pca:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.manifold import trustworthiness`
- **Key imports:** `from sklearn.manifold import trustworthiness`, `from sklearn.model_selection import cross_val_score`, `from sklearn.neighbors import KNeighborsClassifier`
- **Notable calls:** `PCA`, `UMAP`, `cross_val_score`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Vary t-SNE `perplexity` from 5 to 100 — how does the embedding change?
> 2. Replace `make_swiss_roll` with `make_s_curve` — which method handles it best?
> 3. Use `UMAP(n_components=3)` and visualise in 3-D — does it separate the digits?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **Isomap** from scratch (kNN graph -> shortest paths -> MDS).
> 2. Implement **LLE** from scratch and verify against sklearn.
> 3. Implement **t-SNE** from scratch (compute high-D affinities, run gradient
>    descent on the low-D embedding).
> 4. Build a function `compare_embeddings(X, y, methods, ax)` that runs several
>    DR methods and produces a single panel figure.
> 5. Implement the **trustworthiness** metric from scratch.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between t-SNE and UMAP?
> 2. Why is t-SNE not suitable as a general preprocessing step?
> 3. Why does t-SNE preserve local but not global structure?
> 4. What is the role of `perplexity` in t-SNE?
> 5. When would you choose Isomap over PCA?
> 6. What is the trustworthiness metric, and what does a value of 0.95 mean?
> 7. How do non-linear DR methods scale with the number of samples?
> 8. What is a manifold, and what does it mean to assume data lies on one?
