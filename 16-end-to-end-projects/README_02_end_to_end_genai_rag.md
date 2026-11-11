# Code Walkthrough — `02_end_to_end_genai_rag.ipynb`

> Module: **16-end-to-end-projects** | Notebook: **02_end_to_end_genai_rag.ipynb**
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

> # 02 — End-to-End GenAI: Document Q&A RAG
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
> ## 1. Concept Overview — End-to-end GenAI: document Q&A RAG system

> This notebook builds a complete RAG system: ingest PDFs, chunk, embed,
> store in a vector DB, retrieve, and answer questions with an LLM. It's
> the smallest production-quality RAG I can build in one notebook.
> 
> **Beginner track:** follow the 5 steps: ingest → chunk → embed → retrieve
> → answer.
> **Advanced track:** also add re-ranking, hybrid retrieval, and RAGAS
> evaluation.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Same as module 10: embed query $e_q = E(q)$, retrieve top-k by cosine
> similarity, build a prompt with retrieved context, generate answer with
> an LLM.
> 
> New piece: chunking. A document is split into overlapping windows of
> ~500 tokens. Each chunk becomes a row in the vector DB. Chunk size is a
> key hyperparameter — too small loses context, too large dilutes signal.

> ---
> 
> ## 3. Implementation from Scratch — Step 1 — Document ingestion and chunking

> **What this does:** Simulates loading 5 documents (we use plain text
> strings to keep the demo self-contained; in production you'd use
> `PyPDFLoader` or `UnstructuredPDFLoader`). Each document is split into
> chunks of ~100 characters with 20-character overlap.
> 
> **Why it matters:** chunking is the *first* knob of RAG quality. Too small
> = chunks without context; too large = chunks with irrelevant content.
> 
> **What to watch for:** the overlap prevents losing information at chunk
> boundaries. Standard values: chunk size 500-1000 tokens, overlap 50-100.


### Code cell #1
```python
documents = [
    {"id": "doc1", "title": "RAG Fundamentals",
     "text": "Retrieval-Augmented Generation (RAG) combines a retriever with a generator. The retriever finds relevant documents from a corpus. The generator uses the retrieved context to produce an answer. RAG grounds the LLM in up-to-date or private knowledge."},
    {"id": "doc2", "title": "Vector Databases",
     "text": "Vector databases store embeddings and support similarity search. Popular options include FAISS, ChromaDB, Qdrant, and Milvus. Each offers different trade-offs in latency, scale, and ease of use. FAISS is the most established; ChromaDB is the easiest."},
    {"id": "doc3", "title": "Embeddings",
     "text": "Embeddings map text to dense vectors in a high-dimensional space. Sentence-transformers provides high-quality open embeddings. OpenAI text-embedding-3-large is a popular commercial option. Cosine similarity measures semantic closeness between vectors."},
    {"id": "doc4", "title": "Re-ranking",
     "text": "Re-ranking improves retrieval accuracy by using a cross-encoder to score query-document pairs. The cross-encoder sees both query and document jointly, producing a more accurate relevance score. Re-ranking is slower than bi-encoder retrieval but more accurate."},
    {"id": "doc5", "title": "Hybrid Search",
     "text": "Hybrid search combines keyword (BM25) and vector retrieval. The two rankings are fused using Reciprocal Rank Fusion (RRF). Hybrid search catches both exact-match and semantic-match cases. It is the standard for production RAG."},
]

def chunk_text(text, chunk_size=100, overlap=20):
    """Split text into overlapping chunks of approximately chunk_size chars."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["text"], chunk_size=120, overlap=20)
    for i, chunk in enumerate(chunks):
        all_chunks.append({"doc_id": doc["id"], "title": doc["title"],
                            "chunk_id": i, "text": chunk})
print(f"total chunks: {len(all_chunks)}")
print(f"first chunk: {all_chunks[0]['text'][:80]}...")
```

**What this code does:**

- **Headline:** `documents = [`
- **Defines:** def `chunk_text` — Split text into overlapping chunks of approximately chunk_size chars.
- **Notable calls:** `BM25`, `RAG`

> **Takeaway:** Chunking is the first RAG knob. Get the chunk size wrong and
> even the best embedding model can't retrieve well.

> ---
> 
> ## 4. Library Implementation — Step 2 — Embed and store in vector DB

> **What this does:** Builds a mock vector DB (we use random embeddings for
> the demo; in production use `sentence-transformers`). Stores all chunks.
> 
> **Why it matters:** the vector DB is the *retrieval engine* of RAG. Choice
> of embedding model + DB determines the quality ceiling.
> 
> **What to watch for:** in production, you'd use `chromadb` or `faiss`.
> Our mock uses random embeddings — they have no semantic meaning, but the
> code structure is the same.


### Code cell #2
```python
import numpy as np

# Mock embedding function: hash-based deterministic embeddings.
def embed(text, dim=64, seed_offset=0):
    """Hash-based deterministic embedding (mock — no semantic meaning)."""
    seed = (abs(hash(text)) + seed_offset) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)

# Embed all chunks.
for chunk in all_chunks:
    chunk["embedding"] = embed(chunk["text"])

embeddings = np.stack([c["embedding"] for c in all_chunks])
print(f"embeddings shape: {embeddings.shape}")

# Build a mock vector DB: just a numpy array + chunk metadata.
vector_db = {"embeddings": embeddings, "chunks": all_chunks}
print(f"vector DB ready with {len(all_chunks)} chunks")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `embed` — Hash-based deterministic embedding (mock — no semantic meaning).
- **Key imports:** `import numpy as np`

> ---
> 
> ## 5. Dataset Exploration — Step 3 — Retrieve

> **What this does:** Implements retrieval: embed the query, compute cosine
> similarity to all chunks, return top-k. The function we'd use in production
> is the same shape — just `faiss` instead of brute-force numpy.
> 
> **Why it matters:** retrieval is the bottleneck of RAG quality. Bad
> retrieval = bad answer, no matter how good the LLM.
> 
> **What to watch for:** we brute-force over all chunks. In production with
> millions of chunks, you'd use an ANN index (HNSW, IVF).


### Code cell #3
```python
def retrieve(query, db, k=3):
    """Top-k retrieval by cosine similarity."""
    q_emb = embed(query)
    embs = db["embeddings"]
    sims = embs @ q_emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-12)
    top = np.argsort(-sims)[:k]
    return [{"chunk": db["chunks"][i], "similarity": float(sims[i])} for i in top]

results = retrieve("What is RAG?", vector_db, k=3)
print("top-3 retrieved chunks for 'What is RAG?':")
for r in results:
    print(f"  sim={r['similarity']:+.3f}  doc={r['chunk']['doc_id']}  text={r['chunk']['text'][:60]}...")
```

**What this code does:**

- **Headline:** `def retrieve(query, db, k=3):`
- **Defines:** def `retrieve` — Top-k retrieval by cosine similarity.
- **Notable calls:** `RAG`

> **Takeaway:** Retrieval = embed query + cosine search + top-k. The math
> is simple; the engineering is in scaling to millions of chunks.

> ---
> 
> ## 6. Visualization — Step 4 — Generate answer

> **What this does:** Builds the RAG prompt: question + retrieved context +
> instruction. Then calls a mock LLM (we use a function that just returns
> the prompt; in production you'd call OpenAI, Anthropic, or a local LLM).
> 
> **Why it matters:** the prompt template is the *second* RAG knob. A bad
> template (e.g. dumping chunks without structure) hurts more than a weak
> embedding model.
> 
> **What to watch for:** the template explicitly tells the LLM to use *only*
> the context. Without that, the LLM hallucinates from parametric memory.


### Code cell #4
```python
def build_rag_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        f"[{i+1}] {r['chunk']['text']}" for i, r in enumerate(retrieved_chunks)
    )
    return f"""Use the following context to answer the question.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

def mock_llm(prompt):
    # In production: call OpenAI / Anthropic / local LLM.
    return f"(mock answer based on context) "

question = "What is RAG?"
retrieved = retrieve(question, vector_db, k=3)
prompt = build_rag_prompt(question, retrieved)
print("=== RAG PROMPT ===")
print(prompt[:500] + "...")
print("\n=== LLM ANSWER ===")
print(mock_llm(prompt))
```

**What this code does:**

- **Headline:** `def build_rag_prompt(question, retrieved_chunks):`
- **Defines:** def `build_rag_prompt`; def `mock_llm`
- **Notable calls:** `RAG`

> ---
> 
> ## 7. Model Evaluation — Step 5 — End-to-end pipeline + evaluation

> **What this does:** Wraps the 4 steps into a single `rag_answer` function
> and evaluates it on a small Q&A set. We check whether the retrieved chunks
> contain the gold answer (a *faithfulness* proxy).
> 
> **Why it matters:** end-to-end evaluation is the only way to know if your
> RAG actually works. Component-level metrics (recall@k) don't predict final
> answer quality.
> 
> **What to watch for:** real RAG eval uses RAGAS or LangSmith — they measure
> faithfulness, answer relevance, and context precision. We use a simple
> substring check for the demo.


### Code cell #5
```python
def rag_answer(question, db=vector_db, k=3):
    retrieved = retrieve(question, db, k=k)
    prompt = build_rag_prompt(question, retrieved)
    answer = mock_llm(prompt)
    return {"answer": answer, "retrieved_chunks": retrieved, "prompt": prompt}

# Eval set: (question, gold_topic) pairs.
rag_eval = [
    {"q": "What is RAG?",                  "gold_doc": "doc1"},
    {"q": "What vector databases exist?",  "gold_doc": "doc2"},
    {"q": "How do embeddings work?",        "gold_doc": "doc3"},
    {"q": "What is re-ranking?",            "gold_doc": "doc4"},
    {"q": "What is hybrid search?",         "gold_doc": "doc5"},
]

correct = 0
for qa in rag_eval:
    result = rag_answer(qa["q"], k=3)
    retrieved_docs = [r["chunk"]["doc_id"] for r in result["retrieved_chunks"]]
    hit = qa["gold_doc"] in retrieved_docs
    correct += int(hit)
    print(f"  Q: {qa['q'][:30]:32s} gold={qa['gold_doc']} retrieved={retrieved_docs} hit={hit}")
print(f"\nretrieval hit rate: {correct}/{len(rag_eval)} = {correct/len(rag_eval):.0%}")
```

**What this code does:**

- **Headline:** `def rag_answer(question, db=vector_db, k=3):`
- **Defines:** def `rag_answer`
- **Notable calls:** `RAG`

> **Takeaway:** RAG = retrieval + prompt + LLM. End-to-end evaluation is the
> only honest measure. Component metrics are necessary but not sufficient.

> ---
> 
> ## 8. Experiments — Step 6 — Production: FastAPI serving

> **What this does:** Wraps the RAG pipeline in a FastAPI endpoint. Same
> pattern as the churn notebook — a Pydantic request/response, a function
> that calls the pipeline.
> 
> **Why it matters:** a RAG system in a notebook is a demo. A served RAG
> is what users actually use. The endpoint pattern is the standard.
> 
> **What to watch for:** `vector_db` would be loaded at startup, not per
> request. In production, store the embeddings in `chromadb` or `faiss`
> and load the index on startup.


### Code cell #6
```python
from pydantic import BaseModel

class Question(BaseModel):
    question: str
    k: int = 3

class Answer(BaseModel):
    answer: str
    retrieved_docs: list

# In a real app:
# @app.post("/ask", response_model=Answer)
# def ask(q: Question):
#     result = rag_answer(q.question, k=q.k)
#     return Answer(answer=result["answer"],
#                  retrieved_docs=[r["chunk"]["doc_id"] for r in result["retrieved_chunks"]])

# Mock the endpoint call.
sample_q = Question(question="What is RAG?", k=3)
result = rag_answer(sample_q.question, k=sample_q.k)
print(f"input: {sample_q}")
print(f"answer: {result['answer']}")
print(f"retrieved docs: {[r['chunk']['doc_id'] for r in result['retrieved_chunks']]}", )
```

**What this code does:**

- **Headline:** `from pydantic import BaseModel`
- **Defines:** class `Question`; class `Answer`
- **Key imports:** `from pydantic import BaseModel`
- **Notable calls:** `RAG`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Replace mock embeddings with real `sentence-transformers` embeddings.
> 2. Add re-ranking with a cross-encoder.
> 3. Use `chromadb` as the vector DB.
> 4. Add RAGAS evaluation (faithfulness, answer relevance).
> 5. Deploy the FastAPI server with Docker + a real LLM.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Walk through the architecture of a production RAG system.
> 2. How do you choose the chunk size?
> 3. What is the role of re-ranking?
> 4. How do you evaluate a RAG system end-to-end?
> 5. What are the common failure modes of RAG?
> 6. How do you handle multi-hop questions?
> 7. How do you scale RAG to millions of documents?
> 8. How do you handle streaming responses from the LLM?
