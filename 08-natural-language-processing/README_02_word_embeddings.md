# Code Walkthrough — `02_word_embeddings.ipynb`

> Module: **08-natural-language-processing** | Notebook: **02_word_embeddings.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 30
- **Markdown cells:** 24
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

> # 02 — Word Embeddings
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
> ## 1. Concept Overview — Word embeddings

> Bag-of-words ignores the *meaning* of words. Embeddings map each token to a
> dense vector in $\mathbb{R}^d$ where similar words have similar vectors.
> Word2Vec (2013) was the breakthrough; modern transformer embeddings are its
> direct descendant.
> 
> **Beginner track:** focus on the Skip-Gram objective and the intuition of
> the embedding space.
> **Advanced track:** also learn GloVe (matrix factorisation) and contextual
> embeddings (BERT, ELMo).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Skip-Gram Word2Vec**: predict context words $c$ from a target word $t$.
> Maximise:
> $$\frac{1}{T} \sum_{t=1}^{T} \sum_{-w \le j \le w, j \ne 0} \log p(c_{t+j} \mid w_t)$$
> 
> with $p(c \mid t) = \sigma(w_t^\top w_c)$. Trained by SGD with negative sampling.
> 
> **GloVe**: factorise the log-co-occurrence matrix $X$ into $W \cdot V^\top$.
> 
> **Cosine similarity**: $\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a 2-D co-occurrence matrix from a tiny corpus,
> then factorises it with truncated SVD to get word embeddings. This is the
> essence of GloVe — without the GloVe-specific weighting.
> 
> **Why it matters:** SVD-on-co-occurrence is the *original* (1990s) approach
> to embeddings. Word2Vec and GloVe both refine it; SVD-on-co-occurrence is
> the conceptual ancestor.
> 
> **What to watch for:** the embeddings live in 2-D for visualisation. Real
> embeddings are 100-300 dimensional.


### Code cell #1
```python
import numpy as np
from collections import Counter

corpus = "the cat sat on the mat the dog sat on the log cats and dogs are friends".split()
vocab_list = sorted(set(corpus))
vocab_idx = {w: i for i, w in enumerate(vocab_list)}
V = len(vocab_list)

# 1. Build a symmetric co-occurrence matrix with a window of 2.
window = 2
M = np.zeros((V, V), dtype=np.float64)
for i, w in enumerate(corpus):
    for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
        if j != i:
            M[vocab_idx[w], vocab_idx[corpus[j]]] += 1

# 2. Truncated SVD: factorise M = U S V^T, keep top-k singular values.
U, S, Vt = np.linalg.svd(M, full_matrices=False)
k = 2  # 2-D for visualisation
embeddings = U[:, :k] * S[:k]   # scale by sqrt of eigenvalues

print("vocab:", vocab_list)
print("embedding for 'cat':", embeddings[vocab_idx["cat"]].round(3))
print("embedding for 'dog':", embeddings[vocab_idx["dog"]].round(3))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Key imports:** `import numpy as np`, `from collections import Counter`

> **What this does:** Computes cosine similarity between every pair of words
> in the learned embedding space. Words that appear in similar contexts
> should have high cosine similarity.
> 
> **Why it matters:** this is the *test* of any embedding. If your
> embeddings are good, `sim(cat, dog) > sim(cat, log)`.
> 
> **What to watch for:** cosine similarity is invariant to vector magnitude,
> which is what you want for embeddings.


### Code cell #2
```python
def cosine(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)

pairs = [("cat", "dog"), ("cat", "log"), ("cat", "mat"), ("dog", "log"),
         ("cats", "dogs"), ("cats", "are")]
for w1, w2 in pairs:
    if w1 in vocab_idx and w2 in vocab_idx:
        s = cosine(embeddings[vocab_idx[w1]], embeddings[vocab_idx[w2]])
        print(f"sim({w1:5s}, {w2:5s}) = {s:+.3f}")
```

**What this code does:**

- **Headline:** `def cosine(u, v):`
- **Defines:** def `cosine`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Loads pretrained GloVe embeddings (6B, 100d) and tests
> the classic word-similarity analogies. We try to load from a HuggingFace
> cache, with a graceful fallback if not available.
> 
> **Why it matters:** GloVe and Word2Vec embeddings are still the standard
> baseline for many NLP tasks — fast, simple, and surprisingly effective.
> Modern transformers improve on them but at 1000× the compute cost.
> 
> **What to watch for:** GloVe vectors are not centred. `sim(king, queen)`
> is typically ~0.65, not 1.0 — even though the words are related.


### Code cell #3
```python
import numpy as np

# We construct a tiny synthetic embedding matrix to demonstrate the API
# without requiring a 1GB GloVe download.
rng = np.random.default_rng(42)
toy_words = ["king", "queen", "man", "woman", "paris", "france", "rome", "italy"]
# Build a hand-crafted 4D embedding with structure: gender × royal × country × capital.
toy_vectors = {
    "king":   [1.0, 1.0, 0.0, 0.0],
    "queen":  [-1.0, 1.0, 0.0, 0.0],
    "man":    [1.0, 0.0, 0.0, 0.0],
    "woman":  [-1.0, 0.0, 0.0, 0.0],
    "paris":  [0.0, 0.0, 0.0, 1.0],
    "france": [0.0, 0.0, 1.0, 0.0],
    "rome":   [0.0, 0.0, 0.0, 1.0],
    "italy":  [0.0, 0.0, 1.0, 0.0],
}
def cosine(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)

# Classic analogy: king - man + woman = queen
v_king = np.array(toy_vectors["king"])
v_man  = np.array(toy_vectors["man"])
v_woman = np.array(toy_vectors["woman"])
result = v_king - v_man + v_woman
closest = min(toy_vectors, key=lambda w: -cosine(result, np.array(toy_vectors[w])))
print(f"king - man + woman = {closest} (closest by cosine)")

# Top-5 similar words to "king".
sims = sorted(toy_vectors, key=lambda w: -cosine(v_king, np.array(toy_vectors[w])))
print("top-5 closest to 'king':", sims[:5])
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `cosine`
- **Key imports:** `import numpy as np`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small text corpus from a few public-domain
> sentences, then trains a 5-word-context Word2Vec model on it with `gensim`.
> If `gensim` is missing, we fall back to the SVD-on-co-occurrence approach.
> 
> **Why it matters:** `gensim.Word2Vec` is the production-grade way to train
> word embeddings on your own corpus. The API is one line; the trick is
> having enough data.
> 
> **What to watch for:** Word2Vec needs *a lot* of data to learn good
> embeddings — at least 100k tokens. Our toy demo will produce mediocre
> embeddings because the corpus is tiny.


### Code cell #4
```python
corpus_text = [
    "the king ruled the kingdom wisely",
    "the queen ruled alongside the king",
    "the man went to the market",
    "the woman went to the market",
    "paris is the capital of france",
    "rome is the capital of italy",
    "kings and queens are royalty",
    "men and women are people",
] * 50  # oversample for training signal

try:
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=[s.split() for s in corpus_text],
                     vector_size=20, window=3, min_count=1, epochs=50, seed=0)
    print("gensim Word2Vec trained; vocab size:", len(model.wv))
    print("sim(king, queen):", model.wv.similarity("king", "queen"))
except ImportError:
    print("gensim not installed — pip install gensim")
```

**What this code does:**

- **Headline:** `corpus_text = [`
- **Key imports:** `from gensim.models import Word2Vec`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Projects our toy 4-D embeddings to 2-D with PCA and
> plots the words. The classic "king - man + woman = queen" geometry should
> be visible.
> 
> **Why it matters:** the *analogies as parallelograms* property is the most
> famous finding from word embeddings. It works because the embedding space
> linearly encodes semantic relationships.
> 
> **What to watch for:** with only 4 dimensions, PCA preserves 100% of the
> variance. With 300-D real embeddings, PCA to 2-D loses information — use
> t-SNE or UMAP for visualisation.


### Code cell #5
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = list(toy_vectors.keys())
X = np.array([toy_vectors[w] for w in words])
Z = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(Z[:, 0], Z[:, 1], s=80, color="steelblue", edgecolor="white")
for w, (x, y) in zip(words, Z):
    ax.annotate(w, (x, y), fontsize=11, xytext=(5, 5), textcoords="offset points")

# Draw the analogy vector: king - man + woman = queen.
v_king = Z[words.index("king")]
v_man  = Z[words.index("man")]
v_woman = Z[words.index("woman")]
v_queen = Z[words.index("queen")]
ax.arrow(*v_man, *(v_king - v_man), color="red", alpha=0.5, lw=1.5)
ax.arrow(*v_woman, *(v_queen - v_woman), color="red", alpha=0.5, lw=1.5)
ax.set_title("Toy word embeddings (PCA 2-D) — red arrows: king-man and queen-woman")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.decomposition import PCA`
- **Key imports:** `from sklearn.decomposition import PCA`, `import matplotlib.pyplot as plt`
- **Notable calls:** `PCA`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Embeddings are evaluated by:
> 
> - **Word similarity benchmarks** (WordSim-353, SimLex-999) — correlation
>   between cosine similarity and human ratings.
> - **Analogy benchmarks** (Google analogy test set) — `king - man + woman = queen`.
> - **Downstream task performance** — embeddings are good if they help your
>   classifier / sentiment model.

> **What this does:** Evaluates our toy embeddings on a tiny 3-pair word
> similarity benchmark by computing Spearman correlation between cosine
> similarity and human ratings.
> 
> **Why it matters:** this is *exactly* how WordSim-353 works, just on 3 pairs
> instead of 353. The principle is the same.
> 
> **What to watch for:** the correlation here is meaningless with 3 pairs —
> real benchmarks use hundreds.


### Code cell #6
```python
from scipy.stats import spearmanr

# Toy benchmark: word pairs with human similarity scores (0-10).
pairs = [("king", "queen", 9.0),
         ("man", "woman", 8.5),
         ("paris", "france", 8.0),
         ("king", "paris", 1.0),
         ("queen", "man", 1.5),
         ("paris", "rome", 6.5)]

cosines = []
ratings = []
for w1, w2, r in pairs:
    c = cosine(np.array(toy_vectors[w1]), np.array(toy_vectors[w2]))
    cosines.append(c); ratings.append(r)

rho, _ = spearmanr(cosines, ratings)
print(f"Spearman correlation (cosine vs human rating): {rho:.3f}")
for (w1, w2, r), c in zip(pairs, cosines):
    print(f"  {w1:6s} ~ {w2:6s}  human={r:.1f}  cosine={c:+.3f}")
```

**What this code does:**

- **Headline:** `from scipy.stats import spearmanr`
- **Key imports:** `from scipy.stats import spearmanr`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Train Word2Vec on 100k Wikipedia sentences — measure similarity improvement.
> 2. Replace our 4-D toy vectors with real GloVe-50d embeddings (download from Stanford).
> 3. Try `king - man + woman = ?` with real GloVe vectors.
> 4. Compare Word2Vec vs FastText on a word-similarity benchmark.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement Skip-Gram with negative sampling from scratch in PyTorch.
> 2. Implement GloVe from scratch (weighted matrix factorisation).
> 3. Build a sub-word embedding model (like FastText) that handles OOV words.
> 4. Train embeddings on a domain-specific corpus and evaluate them on a
>   domain analogy task.
> 5. Implement a sentence embedding by averaging word embeddings, then by
>   SIF (Smooth Inverse Frequency).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between Word2Vec (Skip-Gram) and GloVe?
> 2. What is negative sampling, and why is it used?
> 3. Why do word embeddings exhibit linear analogies (king - man + woman = queen)?
> 4. What are contextual embeddings, and how do they differ from Word2Vec?
> 5. How would you evaluate a word embedding model?
> 6. What is the effect of embedding dimension on downstream performance?
> 7. What is FastText, and what problem does it solve?
> 8. Why do we use cosine similarity instead of Euclidean distance for embeddings?
