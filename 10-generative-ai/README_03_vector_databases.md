# Code Walkthrough — `03_vector_databases.ipynb`

> Module: **10-generative-ai** | Notebook: **03_vector_databases.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 33
- **Markdown cells:** 26
- **Code cells:** 7

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

> # 03 — Vector Databases
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
> ## 1. Concept Overview — Vector databases

> A vector database stores embeddings and supports *similarity search*: given
> a query vector, return the top-k closest stored vectors. It's the engine
> behind every RAG system, recommender, and image-search pipeline.
> 
> **Beginner track:** understand exact search (brute force) and *approximate*
> search (HNSW, IVF). Most production systems use approximate.
> **Advanced track:** also learn PQ (product quantisation), IVF-PQ, and the
> GPU-accelerated variants (RAFT, FAISS-GPU).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Exact search**: compute $\text{sim}(q, x_i)$ for all $i$, return top-k.
> Time $O(N d)$, exact.
> 
> **Approximate search** (HNSW): build a *hierarchical small-world* graph.
> Walk from a random node to neighbours that are closer to the query, repeat
> at multiple graph levels. Time $O(\log N \cdot d)$, approximate.
> 
> **IVF** (inverted file index): cluster vectors with k-means; search only
> the few clusters whose centroid is closest to the query. Trade recall for
> speed.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements exact (brute-force) similarity search in
> NumPy. Simple, exact, and slow — the baseline against which to compare
> approximate methods.
> 
> **Why it matters:** exact search is what every vector database does at its
> core; approximate methods just trade a little accuracy for a lot of speed.
> Knowing the exact baseline makes it possible to measure the trade-off.
> 
> **What to watch for:** with `N=10_000` vectors of `d=128`, exact search
> takes a few ms in NumPy. With `N=10M`, you need an approximate index.


### Code cell #1
```python
import numpy as np

class ExactVectorDB:
    """Brute-force cosine similarity search."""
    def __init__(self, dim):
        self.dim = dim
        self.vectors = None
    def add(self, vectors):
        # vectors: (N, dim)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        # Pre-compute norms for cosine.
        self.norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12
    def search(self, query, k=5):
        q_norm = np.linalg.norm(query) + 1e-12
        sims = (self.vectors @ query) / (self.norms.ravel() * q_norm)
        top = np.argsort(-sims)[:k]
        return [(i, sims[i]) for i in top]

# Build a tiny DB of 1000 vectors.
rng = np.random.default_rng(0)
db = ExactVectorDB(dim=32)
db.add(rng.standard_normal((1000, 32)))

query = rng.standard_normal(32)
results = db.search(query, k=5)
print("top-5 results (index, cosine sim):")
for i, s in results:
    print(f"  idx={i:4d}  sim={s:+.3f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** class `ExactVectorDB` — Brute-force cosine similarity search.
- **Key imports:** `import numpy as np`

> **What this does:** Implements a tiny IVF (inverted file) index: cluster
> vectors with k-means, then search only the few clusters whose centroids
> are closest to the query.
> 
> **Why it matters:** IVF is the simplest *approximate* index. Understanding
> it teaches the principle of every approximate index: trade recall for
> speed by partitioning the search space.
> 
> **What to watch for:** `n_probe` is the knob. `n_probe=1` is fastest but
> misses more; `n_probe=n_clusters` is exact (and slow). 8-16 probes is a
> common default for ~95% recall.


### Code cell #2
```python
def kmeans(X, k, n_iter=10, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(n_iter):
        # Assign each point to nearest centroid.
        d = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = d.argmin(axis=1)
        # Update centroids.
        for j in range(k):
            if (labels == j).any():
                centroids[j] = X[labels == j].mean(axis=0)
    return centroids, labels

class IVFIndex:
    def __init__(self, n_clusters=10, n_probe=3):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
    def build(self, X):
        self.X = X
        self.centroids, self.labels = kmeans(X, self.n_clusters, seed=0)
        # Inverted lists: cluster_id -> list of vector indices.
        self.inverted = {j: np.where(self.labels == j)[0] for j in range(self.n_clusters)}
    def search(self, query, k=5):
        # Find the n_probe nearest centroids.
        d = np.linalg.norm(self.centroids - query, axis=1)
        probes = np.argsort(d)[:self.n_probe]
        # Gather candidate vectors from probed clusters.
        candidates = np.concatenate([self.inverted[c] for c in probes])
        if len(candidates) == 0:
            return []
        X_cand = self.X[candidates]
        sims = X_cand @ query / (np.linalg.norm(X_cand, axis=1) * np.linalg.norm(query) + 1e-12)
        top = np.argsort(-sims)[:k]
        return [(candidates[i], sims[i]) for i in top]

# Build IVF on the same 1000 vectors.
ivf = IVFIndex(n_clusters=20, n_probe=3)
ivf.build(db.vectors)
ivf_results = ivf.search(query, k=5)
print("IVF top-5 (n_probe=3):")
for i, s in ivf_results:
    print(f"  idx={i:4d}  sim={s:+.3f}")
```

**What this code does:**

- **Headline:** `def kmeans(X, k, n_iter=10, seed=0):`
- **Defines:** def `kmeans`; class `IVFIndex`

> **Takeaway:** An IVF index trades recall for speed by partitioning the
> search space. The `n_probe` knob lets you choose the trade-off.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds a FAISS index three ways: exact (IndexFlatIP),
> IVF (IndexIVFFlat), and HNSW (IndexHNSWFlat). If FAISS isn't installed,
> we skip the demo gracefully.
> 
> **Why it matters:** FAISS is the standard vector database library. Knowing
> the three index types — and when to use each — is the core skill of
> vector-DB engineering.
> 
> **What to watch for:** `IndexFlatIP` uses inner product, which equals
> cosine similarity if vectors are L2-normalised. Always normalise before
> building a `FlatIP` index.


### Code cell #3
```python
try:
    import faiss
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 64)).astype(np.float32)
    faiss.normalize_L2(X)   # in-place normalisation

    # 1. Exact (brute force).
    flat = faiss.IndexFlatIP(64)
    flat.add(X)
    q = rng.standard_normal((1, 64)).astype(np.float32)
    faiss.normalize_L2(q)
    sims, idx = flat.search(q, k=5)
    print("IndexFlatIP top-5 sims:", sims[0].round(3))

    # 2. IVF (approximate, fast).
    quantizer = faiss.IndexFlatIP(64)
    ivf = faiss.IndexIVFFlat(quantizer, 64, nlist=50)
    ivf.train(X)
    ivf.add(X)
    ivf.nprobe = 5   # number of clusters to probe
    sims2, idx2 = ivf.search(q, k=5)
    print("IndexIVFFlat top-5 sims:", sims2[0].round(3))

    # 3. HNSW (graph-based, fast + good recall).
    hnsw = faiss.IndexHNSWFlat(64, M=16)
    hnsw.add(X)
    sims3, idx3 = hnsw.search(q, k=5)
    print("IndexHNSWFlat top-5 sims:", sims3[0].round(3))
except ImportError as e:
    print(f"skipped FAISS demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import faiss`, `import numpy as np`
- **Notable calls:** `faiss`

> **What this does:** Demonstrates `chromadb`, a higher-level vector database
> with built-in persistence and embedding integration. Often used as the
> default in LangChain RAG tutorials.
> 
> **Why it matters:** ChromaDB is the *ergonomic* vector DB — you don't need
> to think about indexing; it picks defaults. Great for prototyping; for
> production at scale you might move to FAISS, Milvus, or Qdrant.
> 
> **What to watch for:** `chromadb` downloads a default embedding model on
> first run. You can also pass your own embeddings if you prefer.


### Code cell #4
```python
try:
    import chromadb
    client = chromadb.Client()
    col = client.create_collection("docs")
    col.add(
        documents=["Python is great for data science.",
                   "Rust is for systems programming.",
                   "Transformers use self-attention."],
        ids=["d1", "d2", "d3"],
    )
    results = col.query(query_texts=["What is Python good for?"], n_results=2)
    print("retrieved docs:", results["documents"][0])
except ImportError:
    print("chromadb not installed — pip install chromadb")
except Exception as e:
    print(f"skipped chromadb demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `import chromadb`
- **Notable calls:** `chromadb`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a 10,000-vector synthetic dataset (1024-dim each)
> and measures how the exact vs IVF indexes scale with dataset size.
> 
> **Why it matters:** the *scaling* behaviour is the most important fact
> about vector search. Exact search is $O(N)$; approximate is $O(\log N \cdot
> n_\text{probe})$.
> 
> **What to watch for:** 1024-dim is a typical embedding dim for modern LLM
> encoders. 10k vectors is small enough to fit in RAM; 100M would need a
> sharded index.


### Code cell #5
```python
import time
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((10000, 1024)).astype(np.float32)
np.linalg.norm(X, axis=1, keepdims=True)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

print(f"dataset shape: {X.shape}, total size: {X.nbytes / 1e6:.1f} MB")

# Time exact search at multiple sizes.
for N in [1000, 5000, 10000]:
    db = ExactVectorDB(dim=1024)
    db.add(X[:N])
    q = rng.standard_normal(1024)
    t0 = time.perf_counter()
    for _ in range(10):
        db.search(q, k=5)
    t1 = time.perf_counter()
    print(f"  N={N:6d}  exact search avg: {(t1-t0)/10*1000:.2f} ms")
```

**What this code does:**

- **Headline:** `import time`
- **Key imports:** `import time`, `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the recall-vs-latency trade-off of our IVF index
> as we vary `n_probe`. As `n_probe` grows, recall approaches 1.0 but latency
> grows.
> 
> **Why it matters:** this *trade-off curve* is the universal picture of any
> approximate vector index. You pick the point on the curve that fits your
> latency budget.
> 
> **What to watch for:** the knee of the curve (here around `n_probe=8`) is
> where you get most of the recall for the least latency. Real systems
> operate near the knee.


### Code cell #6
```python
import matplotlib.pyplot as plt

ivf_test = IVFIndex(n_clusters=20, n_probe=1)
ivf_test.build(db.vectors)
ground_truth = db.search(query, k=5)
gt_ids = set(i for i, _ in ground_truth)

n_probes = [1, 2, 3, 5, 8, 10, 15, 20]
recalls, latencies = [], []
import time
for np_ in n_probes:
    ivf_test.n_probe = np_
    t0 = time.perf_counter()
    for _ in range(10):
        ivf_test.search(query, k=5)
    t1 = time.perf_counter()
    res = ivf_test.search(query, k=5)
    found = len(set(i for i, _ in res) & gt_ids)
    recalls.append(found / 5)
    latencies.append((t1 - t0) / 10 * 1000)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(n_probes, recalls, "o-", color="steelblue", label="recall@5")
ax1.set_xlabel("n_probe"); ax1.set_ylabel("recall", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(n_probes, latencies, "s-", color="orange", label="latency (ms)")
ax2.set_ylabel("latency (ms)", color="orange")
plt.title("IVF recall vs latency trade-off")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import time`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Vector DBs are evaluated by:
> 
> - **Recall@k** vs exact search — the core quality metric.
> - **Latency** at the desired recall — the core performance metric.
> - **Memory footprint** — for billion-scale indexes.
> - **Build time** — how long does it take to construct the index?

> **What this does:** Measures recall@10 for IVF against exact search at
> different `n_probe` settings.
> 
> **Why it matters:** this is the standard benchmark you'd run before
> deploying any vector index. The right `n_probe` is the smallest value that
> gets you above your recall threshold.
> 
> **What to watch for:** we use 100 random queries and average, which is
> what real benchmarks do. Single-query recall is noisy.


### Code cell #7
```python
# Compare IVF recall@10 against exact search, averaged over 100 queries.
rng = np.random.default_rng(1)
queries = rng.standard_normal((100, 32))

for np_ in [1, 3, 8, 20]:
    ivf_test.n_probe = np_
    total_recall = 0
    for q in queries:
        exact_ids = set(i for i, _ in db.search(q, k=10))
        ivf_res = ivf_test.search(q, k=10)
        ivf_ids = set(i for i, _ in ivf_res)
        total_recall += len(exact_ids & ivf_ids) / 10
    print(f"n_probe={np_:2d}  recall@10 = {total_recall/100:.2f}")
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(1)`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Use a real embedding model (`sentence-transformers`) and compare its embeddings
>   to random ones.
> 2. Try FAISS `IndexIVFPQ` (product quantisation) — how much memory does it save?
> 3. Compare FAISS, ChromaDB, and Qdrant on the same dataset.
> 4. Build an HNSW index with `M=8` vs `M=32` — what's the recall difference?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement HNSW from scratch — the graph construction and the search walk.
> 2. Implement product quantisation (PQ) for memory-efficient vector storage.
> 3. Build a distributed vector DB with sharding across multiple machines.
> 4. Implement filtered search: "find vectors matching filter AND similar to query."
> 5. Implement a vector DB benchmark suite (recall + latency + memory).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between exact and approximate nearest-neighbour search?
> 2. How does HNSW work, and why is it faster than brute force?
> 3. What is product quantisation, and what does it trade off?
> 4. When would you use FAISS vs ChromaDB vs Qdrant vs Milvus?
> 5. How does recall@k differ from precision@k for retrieval?
> 6. What is the role of `n_probe` in IVF?
> 7. How would you handle a vector DB with 1 billion vectors?
> 8. What is filtered similarity search, and why is it hard?
