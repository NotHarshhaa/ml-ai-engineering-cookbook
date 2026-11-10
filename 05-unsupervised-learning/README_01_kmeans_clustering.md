# Code Walkthrough — `01_kmeans_clustering.ipynb`

> Module: **05-unsupervised-learning** | Notebook: **01_kmeans_clustering.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 24
- **Markdown cells:** 19
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

> # 01 — K-Means Clustering
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
> ## 1. Concept Overview — K-Means — the canonical clustering algorithm

> Given $K$ and points $\{x_i\}$, K-Means finds centroids $\{\mu_k\}$ and
> assignments $\{z_i\}$ that minimise within-cluster sum of squares (WCSS).
> Despite its age (1967), it is still the default first attempt at clustering
> tabular data.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Objective** (Lloyd's algorithm):
> 
> $$\min_{\{\mu_k\}, \{z_i\}} \sum_{i=1}^{n} \| x_i - \mu_{z_i} \|^2 \quad \text{s.t. } z_i \in \{1, \ldots, K\}$$
> 
> **Lloyd's iterations**: alternate between
> 1. *Assignment*: $z_i = \arg\min_k \|x_i - \mu_k\|^2$
> 2. *Update*: $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$
> 
> Until assignments stop changing. The objective is non-increasing, so
> convergence is guaranteed, but to a local minimum.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ initialisation (better than random)
    idx = [rng.integers(len(X))]
    for _ in range(1, k):
        # k-means++: pick next seed with prob proportional to D(x)^2
        dists = np.min(np.linalg.norm(X[:, None] - X[idx], axis=2), axis=1)
        probs = dists / dists.sum()
        idx.append(rng.choice(len(X), p=probs))
    mus = X[idx].astype(float)
    for _ in range(n_iter):
        # assignment
        z = np.argmin(np.linalg.norm(X[:, None] - mus, axis=2), axis=1)
        # update
        new_mus = np.array([X[z == j].mean(axis=0) if np.any(z == j)
                            else mus[j] for j in range(k)])
        if np.allclose(new_mus, mus): break
        mus = new_mus
    # WCSS
    wcss = sum(np.sum((X[z == j] - mus[j])**2) for j in range(k))
    return mus, z, wcss

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal([0, 0], 0.5, (50, 2)),
    rng.normal([5, 5], 0.5, (50, 2)),
    rng.normal([0, 5], 0.5, (50, 2)),
])
mus, z, wcss = kmeans(X, k=3)
print("centroids:\n", mus.round(2))
print("WCSS:", wcss.round(2))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `kmeans`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6,
                       random_state=0, center_box=(-10, 10))
km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0).fit(X)
print("inertia (WCSS):", round(km.inertia_, 2))
print("labels:", km.labels_[:10])
```

**What this code does:**

- **Headline:** `from sklearn.cluster import KMeans`
- **Key imports:** `from sklearn.cluster import KMeans`, `from sklearn.datasets import make_blobs`, `import matplotlib.pyplot as plt`
- **Notable calls:** `KMeans`, `make_blobs`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> We use the classic *Mall Customers* dataset (age, income, spending score) —
> a small enough problem to inspect every cluster by hand.


### Code cell #3
```python
import pandas as pd
url = "https://raw.githubusercontent.com/AbhilipsaSenapati/Mall-Customer-Segmentation/main/Mall_Customers.csv"
try:
    mall = pd.read_csv(url)
except Exception:
    # fallback synthetic dataset (offline)
    rng = np.random.default_rng(0)
    n = 200
    income = rng.normal(60, 25, n).clip(15, 137)
    score  = rng.normal(50, 25, n).clip(1, 99)
    mall = pd.DataFrame({"CustomerID": np.arange(1, n+1),
                        "Genre": rng.choice(["Male","Female"], n),
                        "Age": rng.integers(18, 70, n),
                        "Annual_Income_k$": income.round(1),
                        "Spending_Score_1_100": score.round(1)})
print("shape:", mall.shape)
mall.head()
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
from sklearn.preprocessing import StandardScaler

features = ["Annual_Income_k$", "Spending_Score_1_100"]
X = StandardScaler().fit_transform(mall[features].values)

km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
mall["cluster"] = km.labels_

plt.figure(figsize=(8, 6))
for c in range(5):
    m = mall.cluster == c
    plt.scatter(mall.loc[m, "Annual_Income_k$"],
                mall.loc[m, "Spending_Score_1_100"],
                label=f"cluster {c}")
plt.xlabel("Annual income (k$)"); plt.ylabel("Spending score (1-100)")
plt.title("K-Means segments (k=5)")
plt.legend(); plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `from sklearn.preprocessing import StandardScaler`
- **Notable calls:** `KMeans`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Clustering is unsupervised — there is no ground truth. Two families of
> metrics help:
> 
> - **WCSS / inertia** — minimise; but it always decreases with $K$.
> - **Silhouette** $s_i = \frac{b - a}{\max(a, b)}$ where $a$ = mean intra-cluster
>   distance and $b$ = mean nearest-cluster distance. Range $[-1, 1]$.
> - **Elbow method** — plot WCSS vs $K$; pick the inflection point.
> - **Davies-Bouldin** — ratio of within-cluster to between-cluster; minimise.


### Code cell #5
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

ks = range(2, 11)
wcss, sil, db = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    db.append(davies_bouldin_score(X, km.labels_))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ks, wcss, "o-"); axes[0].set_title("Elbow (WCSS)"); axes[0].set_xlabel("k")
axes[1].plot(ks, sil, "o-"); axes[1].set_title("Silhouette (higher = better)"); axes[1].set_xlabel("k")
axes[2].plot(ks, db, "o-");  axes[2].set_title("Davies-Bouldin (lower = better)"); axes[2].set_xlabel("k")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.metrics import silhouette_score, davies_bouldin_score`
- **Key imports:** `from sklearn.metrics import silhouette_score`
- **Notable calls:** `KMeans`, `silhouette`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Use `init="random"` instead of `k-means++` and run 10 random seeds — how
>    sensitive is the result to initialisation?
> 2. Generate anisotropic clusters (use `make_blobs` with `transformation`)
>    — does K-Means fail? Why?
> 3. Try `MiniBatchKMeans` and compare quality vs speed on 100k samples.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **k-means++** initialisation from scratch and compare convergence
>    against random init.
> 2. Implement **mini-batch k-means** and benchmark against full k-means.
> 3. Implement **spherical k-means** for cosine similarity.
> 4. Implement the **silhouette coefficient** from scratch.
> 5. Build a clustering pipeline that standardises features, runs K-means, and
>    outputs per-cluster summary statistics for a stakeholder.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why does K-Means converge to a local minimum?
> 2. How does k-means++ initialisation improve on random init?
> 3. Why does K-Means assume spherical clusters, and what goes wrong otherwise?
> 4. What is the elbow method, and what are its failure modes?
> 5. When would you prefer silhouette over WCSS?
> 6. How do you scale K-Means to 100M samples?
> 7. What is the relationship between K-Means and Gaussian Mixture Models?
> 8. How do you choose K when the elbow is not visible?
