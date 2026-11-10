# Code Walkthrough — `02_hierarchical_clustering.ipynb`

> Module: **05-unsupervised-learning** | Notebook: **02_hierarchical_clustering.ipynb**
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

> # 02 — Hierarchical Clustering
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
> ## 1. Concept Overview — Hierarchical clustering — a tree of clusters

> Hierarchical clustering produces a *dendrogram*: a tree whose leaves are
> individual samples and whose root is the whole dataset. **Agglomerative**
> (bottom-up) starts with each sample in its own cluster and merges greedily.
> **Divisive** (top-down) is rarely used in practice.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Linkage criteria define the distance between two clusters:
> 
> - **Single**: $\min_{i \in A, j \in B} d(i, j)$ — chaining-prone.
> - **Complete**: $\max_{i \in A, j \in B} d(i, j)$ — tight clusters.
> - **Average**: $\frac{1}{|A||B|} \sum d(i, j)$ — balanced.
> - **Ward**: minimum-increase-in-WCSS — produces compact spherical clusters.
> 
> Complexity is $O(n^3)$ time / $O(n^2)$ memory — practical only up to a few
> thousand samples without approximate algorithms.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch


### Code cell #1
```python
import numpy as np

def pairwise_dist(X):
    return np.linalg.norm(X[:, None] - X, axis=2)

def agglomerative(X, linkage="single"):
    n = len(X)
    clusters = [[i] for i in range(n)]
    history = []
    D = pairwise_dist(X)
    while len(clusters) > 1:
        # find closest pair of clusters
        best = (None, None, np.inf)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                A, B = clusters[i], clusters[j]
                if linkage == "single":
                    d = np.min(D[A][:, B])
                elif linkage == "complete":
                    d = np.max(D[A][:, B])
                elif linkage == "average":
                    d = np.mean(D[A][:, B])
                if d < best[2]:
                    best = (i, j, d)
        i, j, d = best
        history.append((clusters[i], clusters[j], d))
        clusters.append(clusters[i] + clusters[j])
        del clusters[j]; del clusters[i]
    return history

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0, 0], 0.3, (10, 2)),
               rng.normal([4, 4], 0.3, (10, 2)),
               rng.normal([0, 4], 0.3, (10, 2))])
h = agglomerative(X, linkage="single")
print(f"merged in {len(h)} steps")
print("first 3 merges (cluster sizes -> dist):")
for a, b, d in h[:3]:
    print(f"  sizes {len(a)},{len(b)} -> {d:.3f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `pairwise_dist`; def `agglomerative`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation


### Code cell #2
```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X = StandardScaler().fit_transform(load_iris().data)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, method in zip(axes, ["single", "complete", "average", "ward"]):
    Z = linkage(X, method=method)
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * Z[:, 2].max())
    ax.set_title(f"linkage = {method}")
plt.tight_layout(); plt.show()

# Cut the Ward tree at k=3 clusters.
Z = linkage(X, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")
print("cluster sizes:", np.bincount(labels))
```

**What this code does:**

- **Headline:** `from scipy.cluster.hierarchy import linkage, dendrogram, fcluster`
- **Key imports:** `from scipy.cluster.hierarchy import linkage`, `from sklearn.datasets import load_iris`, `from sklearn.preprocessing import StandardScaler`, `import matplotlib.pyplot as plt`
- **Notable calls:** `load_iris`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration


### Code cell #3
```python
import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
X = StandardScaler().fit_transform(wine.data)
y_true = wine.target
print("shape:", X.shape, " classes:", wine.target_names.tolist())
```

**What this code does:**

- **Headline:** `import pandas as pd`
- **Key imports:** `import pandas as pd`, `from sklearn.datasets import load_wine`
- **Notable calls:** `load_wine`

> ---
> 
> ## 6. Visualization — Visualization


### Code cell #4
```python
from sklearn.decomposition import PCA
import seaborn as sns

X_pca = PCA(n_components=2).fit_transform(X)
Z = linkage(X, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="Set1", edgecolor="white")
axes[0].set_title("True wine classes")
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set1", edgecolor="white")
axes[1].set_title("Ward hierarchical clusters")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.decomposition import PCA`
- **Key imports:** `from sklearn.decomposition import PCA`, `import seaborn as sns`
- **Notable calls:** `PCA`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> When ground-truth labels exist (rare in clustering, but instructive here):
> 
> - **Adjusted Rand Index (ARI)** — corrected-for-chance agreement.
> - **Adjusted Mutual Information (AMI)** — information-theoretic agreement.
> When they do not, use silhouette score or interpret the dendrogram.


### Code cell #5
```python
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

ari = adjusted_rand_score(y_true, labels)
ami = adjusted_mutual_info_score(y_true, labels)
sil = silhouette_score(X, labels)
print(f"ARI = {ari:.3f}  AMI = {ami:.3f}  silhouette = {sil:.3f}")
```

**What this code does:**

- **Headline:** `from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette...`
- **Key imports:** `from sklearn.metrics import adjusted_rand_score`
- **Notable calls:** `silhouette`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Cut the dendrogram at $k=4$ instead of $3$ — what does ARI say?
> 2. Switch linkage to `single` — does ARI drop on this dataset? Why?
> 3. Replace Euclidean distance with Manhattan (`metric="cityblock"`) — any change?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement **divisive hierarchical clustering** from scratch.
> 2. Implement the **Lance-Williams update formula** so the merge step is $O(n^2)$.
> 3. Use **scipy.spatial.distance.pdist** with `cosine` and cluster text TF-IDF vectors.
> 4. Implement **DBSCAN** from scratch and compare against hierarchical on a
>    non-spherical cluster dataset (`make_moons`).
> 5. Implement a function that recommends a `cut_threshold` automatically from
>    the largest gap in the dendrogram.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between single, complete, average, and Ward linkage?
> 2. Why is hierarchical clustering $O(n^3)$ time, and how do you make it scalable?
> 3. Why does single linkage suffer from chaining?
> 4. How do you choose the number of clusters from a dendrogram?
> 5. When would you prefer hierarchical over K-Means?
> 6. What is the ARI, and what does a value of 0 mean?
> 7. Why is standardisation important before hierarchical clustering?
> 8. What is the difference between agglomerative and divisive clustering?
