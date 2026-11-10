# Code Walkthrough — `02_rag_fundamentals.ipynb`

> Module: **10-generative-ai** | Notebook: **02_rag_fundamentals.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 31
- **Markdown cells:** 25
- **Code cells:** 6

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

> # 02 — RAG Fundamentals
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
> ## 1. Concept Overview — Retrieval-Augmented Generation (RAG)

> RAG augments an LLM with a *retrieval* step: before answering, the system
> fetches relevant documents from a vector database and includes them in the
> prompt. This grounds the answer in up-to-date or private knowledge.
> 
> **Beginner track:** focus on the three-step pipeline — embed, retrieve,
> generate. Everything else is variations on this.
> **Advanced track:** also learn hybrid retrieval (BM25 + vector),
> re-ranking, chunking strategies, and evaluation (RAGAS, faithfulness).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Given a query $q$ and a corpus $\{d_1, \ldots, d_N\}$:
> 
> 1. Embed: $e_q = E(q)$, $e_i = E(d_i)$ with an embedding model $E$.
> 2. Retrieve: $k$ nearest neighbours by cosine similarity
>    $\text{sim}(e_q, e_i) = \frac{e_q \cdot e_i}{\|e_q\| \|e_i\|}$.
> 3. Generate: $\text{LLM}(\text{prompt}(q, d_{i_1}, \ldots, d_{i_k}))$.
> 
> The LLM sees only the retrieved context — its parametric memory is
> augmented by external knowledge.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a tiny RAG system from scratch in NumPy:
> random embedding vectors for 5 documents, cosine-similarity retrieval,
> and a mock generator that prepends the retrieved context to the prompt.
> 
> **Why it matters:** seeing RAG at this scale removes the magic — it's
> literally `embed → cosine search → stuff into prompt`. Production RAG just
> scales each step.
> 
> **What to watch for:** real embeddings come from a model like
> `sentence-transformers/all-MiniLM-L6-v2`. Our random embeddings can't
> do real semantic similarity — they're a placeholder.


### Code cell #1
```python
import numpy as np

# 5 mock documents, each embedded as a random 8-dim vector.
documents = [
    {"id": 1, "text": "Python is a high-level programming language.", "topic": "python"},
    {"id": 2, "text": "Rust is a systems programming language focused on safety.", "topic": "rust"},
    {"id": 3, "text": "Transformers use self-attention for sequence modelling.", "topic": "ml"},
    {"id": 4, "text": "Pandas is a Python library for tabular data analysis.", "topic": "python"},
    {"id": 5, "text": "CNNs are used for image classification tasks.", "topic": "ml"},
]

# Mock embedding: hash the text to a fixed-size vector. Deterministic.
def mock_embed(text, dim=8, seed=0):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim)

# Pre-compute all document embeddings.
for doc in documents:
    doc["embedding"] = mock_embed(doc["text"])

def retrieve(query, k=2):
    q_emb = mock_embed(query)
    sims = [(d, (q_emb @ d["embedding"]) / (np.linalg.norm(q_emb) * np.linalg.norm(d["embedding"]) + 1e-12))
            for d in documents]
    sims.sort(key=lambda x: -x[1])
    return sims[:k]

results = retrieve("Tell me about Python", k=2)
for doc, sim in results:
    print(f'  sim={sim:+.3f}  id={doc["id"]}  text={doc["text"][:50]}')
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `mock_embed`; def `retrieve`
- **Key imports:** `import numpy as np`

> **What this does:** Mocks the *generation* step — we concatenate the
> retrieved context and the query into a prompt template. The mock generator
> just returns the prompt; a real LLM would complete it.
> 
> **Why it matters:** the RAG prompt template is the most under-appreciated
> part of the pipeline. A bad template (e.g. just dumping the chunks without
> formatting) hurts LLM accuracy more than a weak retrieval model.
> 
> **What to watch for:** the template explicitly tells the LLM to *only* use
> the provided context. Without this, the LLM may hallucinate from its
> parametric memory.


### Code cell #2
```python
def rag_prompt(query, retrieved_docs):
    context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(retrieved_docs))
    return f"""Use the following context to answer the question.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

def mock_generate(prompt):
    # A real LLM would complete the prompt; we just show we built it.
    return f"[mock LLM output] would follow from this prompt:\n{prompt}"

query = "What programming languages are mentioned?"
retrieved = retrieve(query, k=2)
retrieved_docs = [d for d, _ in retrieved]
prompt = rag_prompt(query, retrieved_docs)
print(mock_generate(prompt))
```

**What this code does:**

- **Headline:** `def rag_prompt(query, retrieved_docs):`
- **Defines:** def `rag_prompt`; def `mock_generate`

> **Takeaway:** RAG is a 3-line pipeline: embed → retrieve → prompt. The
> engineering is in the *details*: chunk size, embedding model, retrieval
> method, prompt template.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds a real RAG pipeline with `sentence-transformers`
> for embeddings and FAISS for retrieval. If either library is missing, we
> skip the demo gracefully.
> 
> **Why it matters:** `sentence-transformers` + FAISS is the *standard* RAG
> stack in 2024-2025. Knowing it cold means you can build a working RAG
> system in an afternoon.
> 
> **What to watch for:** `all-MiniLM-L6-v2` is 22M params, runs on CPU, and
> produces 384-dim embeddings. For production you might prefer
> `bge-large-en-v1.5` or OpenAI's `text-embedding-3-large`.


### Code cell #3
```python
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    corpus = [
        "Python is a popular programming language for data science.",
        "Rust is a systems language that emphasises memory safety.",
        "Transformers are neural networks based on self-attention.",
        "Pandas provides dataframes for Python data analysis.",
        "CNNs are commonly used for image classification.",
    ]
    embeddings = embedder.encode(corpus, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine since normalised
    index.add(embeddings)

    query = "What language is used for data analysis?"
    q_emb = embedder.encode([query], normalize_embeddings=True)
    sims, idx = index.search(q_emb, k=2)

    for rank, (sim, i) in enumerate(zip(sims[0], idx[0])):
        print(f"  rank {rank+1} sim={sim:.3f}  -> {corpus[i][:60]}")
except ImportError as e:
    print(f"skipped FAISS/sentence-transformers demo ({e})")
    print("(install: pip install sentence-transformers faiss-cpu)")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from sentence_transformers import SentenceTransformer`, `import faiss`, `import numpy as np`
- **Notable calls:** `sentence_transformers`, `faiss`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small Q&A dataset by treating each of 10
> documents as a chunk and pairing each with a question. We'll use it for the
> RAG evaluation in section 7.
> 
> **Why it matters:** RAG evaluation needs (question, gold_doc) pairs to
> compute retrieval metrics (recall@k). Without a labelled set, you can't
> measure whether your retrieval is actually working.
> 
> **What to watch for:** the question must be paraphrased — *not* a literal
> substring of the document. Otherwise retrieval is trivially easy and you
> can't tell good vs bad embedding models apart.


### Code cell #4
```python
rag_eval = [
    {"question": "What language is used for data science?", "gold_topic": "python"},
    {"question": "Which language focuses on memory safety?", "gold_topic": "rust"},
    {"question": "What neural architecture is based on attention?", "gold_topic": "ml"},
    {"question": "What Python library handles dataframes?", "gold_topic": "python"},
    {"question": "What is used for image classification?", "gold_topic": "ml"},
]
print(f"eval set size: {len(rag_eval)}")
for qa in rag_eval:
    print(f"  Q: {qa['question']}  -> expected topic: {qa['gold_topic']}")
```

**What this code does:**

- **Headline:** `rag_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the cosine similarity matrix between the 5
> documents and 5 queries. A good embedding should show a clear block
> structure (queries match their gold documents).
> 
> **Why it matters:** the similarity heatmap is the most direct visual check
> of whether embeddings are doing their job. If the heatmap is uniform,
> your embeddings are bad.
> 
> **What to watch for:** with mock (random) embeddings, the heatmap is
> uniform. With real sentence-transformer embeddings, you'd see a clear
> diagonal pattern.


### Code cell #5
```python
# Build similarity matrix using mock embeddings.
queries = [qa["question"] for qa in rag_eval]
q_embs = np.array([mock_embed(q) for q in queries])
d_embs = np.array([d["embedding"] for d in documents])

# Cosine similarity matrix: rows = queries, cols = documents.
S = q_embs @ d_embs.T
for i in range(S.shape[0]):
    S[i] /= (np.linalg.norm(q_embs[i]) * np.linalg.norm(d_embs, axis=1) + 1e-12)

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.imshow(S, cmap="viridis", aspect="auto")
plt.colorbar(label="cosine similarity")
plt.xticks(range(len(documents)), [d["topic"][:8] for d in documents], rotation=45)
plt.yticks(range(len(queries)), [f"Q{i+1}" for i in range(len(queries))])
plt.title("Query × document similarity matrix (mock embeddings)")
plt.tight_layout(); plt.show()
print("(with mock embeddings, the matrix is uniform — real embeddings would show structure)")
```

**What this code does:**

- **Headline:** `queries = [qa["question"] for qa in rag_eval]`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> RAG is evaluated on two axes:
> 
> - **Retrieval quality** — recall@k, precision@k, MRR.
> - **Generation quality** — faithfulness (no hallucination), answer
>   relevance, context utilisation.
> 
> Tools: `RAGAS` (Python library) and `LangSmith` (LangChain's eval platform).

> **What this does:** Computes recall@k and MRR (mean reciprocal rank) for
> our mock retrieval on the eval set. Recall@k = fraction of queries where
> the gold document is in the top-k retrieved.
> 
> **Why it matters:** these two metrics are the standard retrieval metrics.
> If you report only one, you'll miss the picture — recall@k rewards breadth,
> MRR rewards ranking quality.
> 
> **What to watch for:** MRR is `1/rank_of_first_gold`, averaged over
> queries. A perfect ranker scores 1.0.


### Code cell #6
```python
def recall_at_k(retrieved_ids, gold_id, k):
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0

def reciprocal_rank(retrieved_ids, gold_id):
    for i, rid in enumerate(retrieved_ids):
        if rid == gold_id:
            return 1.0 / (i + 1)
    return 0.0

# Map topics to document IDs for the eval.
topic_to_id = {d["topic"]: d["id"] for d in documents}

recalls, rrs = [], []
for qa in rag_eval:
    gold_id = topic_to_id[qa["gold_topic"]]
    retrieved = retrieve(qa["question"], k=5)
    retrieved_ids = [d["id"] for d, _ in retrieved]
    recalls.append(recall_at_k(retrieved_ids, gold_id, k=3))
    rrs.append(reciprocal_rank(retrieved_ids, gold_id))

print(f"recall@3 (mock embeddings): {np.mean(recalls):.2f}")
print(f"MRR       (mock embeddings): {np.mean(rrs):.2f}")
print("(low scores expected with random embeddings)")
```

**What this code does:**

- **Headline:** `def recall_at_k(retrieved_ids, gold_id, k):`
- **Defines:** def `recall_at_k`; def `reciprocal_rank`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace mock embeddings with real `sentence-transformers` — how much does recall improve?
> 2. Try `chunk_size=200` vs `chunk_size=500` characters — what's the trade-off?
> 3. Retrieve top-3 vs top-5 vs top-10 — when does more context stop helping?
> 4. Add a re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — does it improve MRR?

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement chunking by sentences, by tokens, and by markdown headers.
> 2. Build a hybrid retriever (BM25 + vector) with reciprocal rank fusion.
> 3. Implement a re-ranker using a cross-encoder.
> 4. Add citation tracking: which retrieved chunk did the answer come from?
> 5. Implement an evaluation pipeline using RAGAS or LangSmith.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What problem does RAG solve that fine-tuning doesn't?
> 2. How do you choose an embedding model for RAG?
> 3. What is the trade-off between chunk size and retrieval accuracy?
> 4. Why use cosine similarity instead of dot product for retrieval?
> 5. What is recall@k, and why is it more informative than top-1 accuracy?
> 6. What is the difference between a bi-encoder and a cross-encoder?
> 7. How would you handle a multi-hop question in RAG?
> 8. What is hallucination, and how does RAG reduce it?
