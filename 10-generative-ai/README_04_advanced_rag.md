# Code Walkthrough — `04_advanced_rag.ipynb`

> Module: **10-generative-ai** | Notebook: **04_advanced_rag.ipynb**
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

> # 04 — Advanced RAG
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
> ## 1. Concept Overview — Advanced RAG techniques

> Plain RAG (embed → retrieve → generate) gets you ~70% of the way. The
> remaining 30% comes from *advanced* techniques: hybrid retrieval,
> re-ranking, query rewriting, multi-hop, and evaluation.
> 
> **Beginner track:** focus on hybrid retrieval (BM25 + vector) and
> re-ranking with a cross-encoder. Those two upgrades give the biggest bang.
> **Advanced track:** also learn HyDE (hypothetical document embeddings),
> step-back prompting, and self-RAG.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Hybrid retrieval**: combine BM25 (keyword) score $s_\text{BM25}(q, d)$
> with vector (semantic) score $s_\text{vec}(q, d)$. Reciprocal rank fusion:
> $$\text{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + r(d)}$$
> 
> **Re-ranking**: take top-N from first-stage retrieval, score each with a
> cross-encoder (a transformer that takes (query, doc) jointly).
> 
> **HyDE**: generate a hypothetical answer with the LLM, embed it, retrieve
> against that — the *imagined* answer often matches real documents better
> than the question.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements BM25 from scratch — the classic keyword
> retrieval algorithm. We score documents by how often each query term
> appears, normalised by document length.
> 
> **Why it matters:** BM25 is still the strongest *keyword* baseline. Real
> RAG systems combine BM25 (catches exact-match terms) with vector retrieval
> (catches semantic match). Knowing BM25 is knowing half of hybrid retrieval.
> 
> **What to watch for:** `k1` and `b` are BM25's hyperparameters. `k1=1.5`
> and `b=0.75` are the standard defaults from the 1994 paper.


### Code cell #1
```python
import math
from collections import Counter

class BM25Scratch:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [d.lower().split() for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avg_len = sum(len(d) for d in self.docs) / self.N
        self.doc_lens = [len(d) for d in self.docs]
        self.df = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
    def idf(self, t):
        return math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
    def score(self, query, doc_idx):
        d = self.docs[doc_idx]
        tf = Counter(d)
        score = 0.0
        for t in query.lower().split():
            if t not in tf: continue
            idf = self.idf(t)
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * self.doc_lens[doc_idx] / self.avg_len)
            score += idf * num / den
        return score
    def search(self, query, k=3):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

docs = [
    "Python is a popular programming language",
    "The python is a large snake found in tropical regions",
    "Rust is a systems programming language",
    "The transformer architecture revolutionised NLP",
]
bm25 = BM25Scratch(docs)
print("BM25 results for 'python language':")
for i, s in bm25.search("python language", k=3):
    print(f"  score={s:.3f}  doc={i}  -> {docs[i][:60]}")
```

**What this code does:**

- **Headline:** `import math`
- **Defines:** class `BM25Scratch`
- **Key imports:** `import math`, `from collections import Counter`
- **Notable calls:** `BM25`

> **Takeaway:** BM25 catches exact keyword matches; vector retrieval catches
> semantics. Combining them gives you the best of both worlds.

> **What this does:** Implements Reciprocal Rank Fusion (RRF): combine the
> rankings from BM25 and vector retrieval into a single ranking without
> needing to normalise scores across very different distributions.
> 
> **Why it matters:** RRF is the standard way to combine multiple retrieval
> rankings. It's simple, parameter-light, and works surprisingly well.
> 
> **What to watch for:** `k=60` is the conventional RRF constant. Lower `k`
> favours top ranks; higher `k` makes the fusion smoother.


### Code cell #2
```python
def rrf(rankings, k=60):
    """rankings: list of ranked lists (each is a list of doc_ids in order)."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

# Mock rankings from BM25 and vector retrieval.
bm25_ranking = [0, 1, 2, 3]      # doc ids in BM25 order
vector_ranking = [0, 2, 3, 1]    # doc ids in vector order

fused = rrf([bm25_ranking, vector_ranking])
print("RRF fused ranking:")
for doc_id, score in fused:
    print(f"  doc={doc_id}  rrf_score={score:.4f}  -> {docs[doc_id][:50]}")
```

**What this code does:**

- **Headline:** `def rrf(rankings, k=60):`
- **Defines:** def `rrf` — rankings: list of ranked lists (each is a list of doc_ids in order).
- **Notable calls:** `BM25`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Demonstrates hybrid retrieval with the `rank_bm25`
> library and `sentence-transformers` for vector search, fused with RRF.
> If either library is missing, we skip the demo.
> 
> **Why it matters:** this is the standard hybrid-retrieval stack. It's the
> default in production RAG systems where exact keyword match matters (legal,
> medical, code-search).
> 
> **What to watch for:** `rank_bm25` is the standard BM25 library — a
> simple, pure-Python implementation with no native dependencies.


### Code cell #3
```python
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    corpus = [
        "Python is a popular programming language for data science.",
        "Rust is a systems language focused on memory safety.",
        "Transformers use self-attention for sequence modelling.",
        "Pandas is a Python library for tabular data analysis.",
        "CNNs are used for image classification tasks.",
    ]
    tokenised = [c.lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenised)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = embedder.encode(corpus, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    query = "What Python library handles dataframes?"
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_rank = (-bm25_scores).argsort().tolist()

    q_emb = embedder.encode([query], normalize_embeddings=True)
    _, vec_idx = index.search(q_emb, k=len(corpus))
    vec_rank = vec_idx[0].tolist()

    fused = rrf([bm25_rank, vec_rank])
    print("Hybrid (RRF) ranking:")
    for i, s in fused[:3]:
        print(f"  rrf_score={s:.4f}  -> {corpus[i][:60]}")
except Exception as e:
    print(f"skipped hybrid demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from rank_bm25 import BM25Okapi`, `from sentence_transformers import SentenceTransformer`, `import faiss`, `import numpy as np`
- **Notable calls:** `sentence_transformers`, `faiss`, `BM25`

> **What this does:** Demonstrates re-ranking: take the top-5 from hybrid
> retrieval and re-score them with a cross-encoder. The cross-encoder sees
> (query, doc) jointly, so it's much more accurate than the bi-encoder used
> for retrieval.
> 
> **Why it matters:** re-ranking is the single biggest accuracy boost you can
> give a RAG system without changing the retrieval index. A small
> cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks 5 docs in <50 ms.
> 
> **What to watch for:** re-rankers are too slow to apply to the whole corpus
> — that's why you retrieve top-N with the fast bi-encoder, then re-rank
> those N with the slow cross-encoder.


### Code cell #4
```python
try:
    from sentence_transformers import CrossEncoder
    # A small cross-encoder trained on MS-MARCO passage ranking.
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, corpus[i]) for i, _ in fused[:5]]
    scores = reranker.predict(pairs)
    reranked = sorted(zip([i for i, _ in fused[:5]], scores), key=lambda x: -x[1])
    print("Re-ranked top-5:")
    for i, s in reranked:
        print(f"  score={s:.3f}  -> {corpus[i][:60]}")
except Exception as e:
    print(f"skipped re-ranking demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from sentence_transformers import CrossEncoder`
- **Notable calls:** `sentence_transformers`, `CrossEncoder`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a slightly larger RAG eval set with hard
> negative examples — questions where a keyword-only retrieval would fail.
> We'll use it to show that hybrid retrieval beats pure BM25.
> 
> **Why it matters:** hard negatives are what make a RAG benchmark useful.
> Without them, you can't distinguish good retrieval from trivially-good
> retrieval.
> 
> **What to watch for:** the question "What does the GIL prevent?" would
> be matched poorly by BM25 (no document literally says "GIL prevent") but
> well by vector retrieval (semantic match).


### Code cell #5
```python
hard_eval = [
    {"q": "What does the GIL prevent?", "gold_id": 0,
     "doc": "Python has a Global Interpreter Lock that prevents true multi-threading."},
    {"q": "Which model uses self-attention?", "gold_id": 1,
     "doc": "The transformer architecture relies on self-attention instead of recurrence."},
    {"q": "What language is safest for systems?", "gold_id": 2,
     "doc": "Rust enforces memory safety at compile time, eliminating many C++ bugs."},
    {"q": "How do CNNs detect objects?", "gold_id": 3,
     "doc": "Convolutional neural networks learn spatial features through convolutions."},
    {"q": "What is Pandas good for?", "gold_id": 4,
     "doc": "Pandas is a Python library used for data manipulation and analysis."},
]
docs = [d["doc"] for d in hard_eval]
print(f"hard eval size: {len(hard_eval)}")
for d in hard_eval[:3]:
    print(f"  Q: {d['q']}")
    print(f"  -> {d['doc']}")
```

**What this code does:**

- **Headline:** `hard_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Compares recall@3 of three retrieval methods on our
> hard eval set: BM25-only, vector-only, and hybrid (BM25 + vector via RRF).
> 
> **Why it matters:** this is the *experiment* that justifies the complexity
> of hybrid retrieval. The point of building it is to show that hybrid
> wins on hard negatives.
> 
> **What to watch for:** on easy datasets, the three methods are similar.
> The hybrid advantage shows on *hard* benchmarks — that's why you need a
> hard eval set.


### Code cell #6
```python
import matplotlib.pyplot as plt

# Mock results (real numbers would require running the live demo).
methods = ["BM25 only", "Vector only", "Hybrid (BM25 + Vector)"]
recall_at_3 = [0.40, 0.60, 0.80]   # typical for hard negatives

plt.figure(figsize=(7, 4))
bars = plt.bar(methods, recall_at_3, color=["#e74c3c", "#3498db", "#27ae60"],
               edgecolor="white")
plt.ylim(0, 1.0)
plt.ylabel("recall@3 on hard negatives")
plt.title("Hybrid retrieval beats single-method on hard queries")
for bar, r in zip(bars, recall_at_3):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{r:.0%}",
             ha="center", fontsize=10)
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`
- **Notable calls:** `BM25`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Advanced RAG evaluation uses frameworks like **RAGAS** which compute:
> 
> - **Faithfulness** — is the answer supported by retrieved context?
> - **Answer relevance** — does the answer address the question?
> - **Context precision** — are the retrieved chunks relevant?
> - **Context recall** — is the gold answer recoverable from the chunks?

> **What this does:** Implements a tiny *faithfulness* scorer: did the answer
> claim anything not in the retrieved context? We use a simple overlap-based
> heuristic (a real system would use an LLM-as-judge).
> 
> **Why it matters:** faithfulness is the metric that catches hallucinations.
> If your RAG system has high answer-relevance but low faithfulness, it's
> convincingly wrong.
> 
> **What to watch for:** this heuristic is intentionally crude. Real RAGAS
> uses an LLM to check whether each claim in the answer is supported by the
> context.


### Code cell #7
```python
def faithfulness_heuristic(answer, context):
    """Crude overlap-based faithfulness: fraction of answer words found in context."""
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    if not answer_tokens:
        return 1.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)

# Mock answer and context.
context = "Python has a Global Interpreter Lock that prevents true multi-threading."
faithful_answer   = "The GIL prevents true multi-threading in Python."
unfaithful_answer = "The GIL prevents Python from running on Windows."

print(f"faithful answer faithfulness: {faithfulness_heuristic(faithful_answer, context):.2f}")
print(f"unfaithful answer faithfulness: {faithfulness_heuristic(unfaithful_answer, context):.2f}")
```

**What this code does:**

- **Headline:** `def faithfulness_heuristic(answer, context):`
- **Defines:** def `faithfulness_heuristic` — Crude overlap-based faithfulness: fraction of answer words found in context.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Try HyDE: generate a hypothetical answer with an LLM, embed it, retrieve against that.
> 2. Implement query rewriting: ask the LLM to expand the query into 3 variants before retrieval.
> 3. Implement *multi-hop* RAG: retrieve, generate a follow-up question, retrieve again.
> 4. Try `bge-reranker-large` instead of the MiniLM re-ranker — does MRR improve?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a full RAGAS evaluation (faithfulness, answer relevance, context precision/recall).
> 2. Implement HyDE and measure recall improvement on a hard eval set.
> 3. Build a *self-RAG* pipeline: the LLM decides whether to retrieve.
> 4. Implement parent-document retrieval: chunk small, return large.
> 5. Build a multi-modal RAG system that retrieves both text and images.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why does hybrid retrieval (BM25 + vector) outperform either alone?
> 2. What is reciprocal rank fusion, and why does it work?
> 3. What is re-ranking, and when is it worth the latency cost?
> 4. What is HyDE, and what problem does it solve?
> 5. How does RAGAS evaluate a RAG pipeline?
> 6. What is *faithfulness* in RAG evaluation, and why does it matter?
> 7. How would you handle multi-hop reasoning in a RAG system?
> 8. What are *hard negatives*, and why are they important for retrieval eval?
