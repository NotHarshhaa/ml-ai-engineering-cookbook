# Code Walkthrough — `01_text_preprocessing_tokenization.ipynb`

> Module: **08-natural-language-processing** | Notebook: **01_text_preprocessing_tokenization.ipynb**
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

> # 01 — Text Preprocessing & Tokenisation
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
> ## 1. Concept Overview — Text preprocessing & tokenisation

> Every NLP pipeline starts with the same three steps: *clean* the text,
> *tokenise* it into units, and *index* those units into integers. The choices
> you make here ripple through every downstream model.
> 
> **Beginner track:** focus on whitespace tokenisation, regex cleaning, and
> vocabulary building — that's enough for bag-of-words models.
> **Advanced track:** also learn sub-word tokenisers (BPE, WordPiece, SentencePiece)
> and how HuggingFace's `tokenizers` library handles them.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A tokeniser is a function $\tau: \text{string} \to \mathbb{Z}^L$ mapping a
> string of length $L$ tokens to integer IDs. The vocabulary $V$ is a bijection
> between token types and IDs. A bag-of-words vector is a sparse histogram over $V$:
> $$\text{bow}(x) = \sum_{t \in \tau(x)} e_{\text{id}(t)}$$

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Builds a tiny NLP pipeline by hand — regex-based text
> cleaning, whitespace tokeniser, vocabulary mapping, and a bag-of-words
> encoder. The result is the same data structure that 2000s-era NLP used.
> 
> **Why it matters:** Understanding the from-scratch version is what makes
> the HuggingFace `tokenizers` library make sense. Sub-word tokenisers just
> swap the *what is a token* question; everything else stays the same.
> 
> **What to watch for:** `re.sub(r"\s+", " ", text)` normalises all whitespace
> (tabs, newlines, multiple spaces) to a single space. Without it, the
> vocabulary contains duplicate tokens like "the" and "the\n".


### Code cell #1
```python
import re
from collections import Counter

def clean_text(text):
    text = text.lower()                           # normalise case
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # strip punctuation
    text = re.sub(r"\s+", " ", text)                # collapse whitespace
    return text.strip()

def tokenize(text):
    """Whitespace tokeniser — the simplest possible."""
    return text.split()

class Vocab:
    def __init__(self, tokens, min_freq=1, max_size=10000):
        counter = Counter(tokens)
        # Always include <pad> and <unk> at IDs 0 and 1.
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for tok, freq in counter.most_common(max_size):
            if freq >= min_freq:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
    def __len__(self): return len(self.itos)
    def encode(self, tokens):
        return [self.stoi.get(t, 1) for t in tokens]   # <unk> for OOV

corpus = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are friends.",
]
tokens = [tokenize(clean_text(d)) for d in corpus]
flat = [t for doc in tokens for t in doc]
vocab = Vocab(flat, min_freq=1)
print("vocab size:", len(vocab))
print("first 12 tokens:", vocab.itos[:12])
print("encoded doc 0:", vocab.encode(tokens[0]))
```

**What this code does:**

- **Headline:** `import re`
- **Defines:** def `clean_text`; def `tokenize` — Whitespace tokeniser — the simplest possible.; class `Vocab`
- **Key imports:** `import re`, `from collections import Counter`

> **Takeaway:** Every NLP pipeline is *clean → tokenise → encode → embed →
> model*. The first three steps are mechanical; the creativity is in the
> last two.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-implements the same pipeline with `nltk` and
> `sklearn.feature_extraction.text` — the production-grade libraries.
> 
> **Why it matters:** The from-scratch version is for learning. For real
> work, use `nltk` (or `spaCy`) for tokenisation and `sklearn` for bag-of-words.
> They handle edge cases you would otherwise miss.
> 
> **What to watch for:** `CountVectorizer(stop_words="english")` removes
> English stop words automatically. That's usually good for topic modelling,
> usually bad for sentiment analysis (where stop words like "not" matter).


### Code cell #2
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=True, stop_words="english",
                              token_pattern=r"[a-z]+", min_df=1)
X = vectorizer.fit_transform(corpus)
print("vocab size:", len(vectorizer.vocabulary_))
print("feature names (first 10):", vectorizer.get_feature_names_out()[:10])
print("doc-term matrix shape:", X.shape)
print("row 0 (non-zero entries):")
for i, j in zip(*X[0].nonzero()):
    print(f"  token={vectorizer.get_feature_names_out()[j]:10s} count={X[0, i]}", end="\n")
```

**What this code does:**

- **Headline:** `from sklearn.feature_extraction.text import CountVectorizer`
- **Key imports:** `from sklearn.feature_extraction.text import CountVectorizer`

> **What this does:** Uses a HuggingFace `BertTokenizer` to show what a
> modern sub-word tokeniser does. The same sentence becomes a sequence of
> sub-word units — handles out-of-vocabulary words by composition.
> 
> **Why it matters:** every transformer-based model (BERT, GPT, T5) uses a
> sub-word tokeniser. The reason: sub-words handle rare words ("unbelievably"
> → "un", "##believe", "##ably") without exploding the vocabulary.
> 
> **What to watch for:** `##` is BERT's continuation marker — it means "this
> sub-word continues the previous one" (no space before it).


### Code cell #3
```python
try:
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "Unbelievably, transformer tokenisers handle rare words."
    encoded = tok.encode(text)
    decoded = tok.convert_ids_to_tokens(encoded)
    print("input:", text)
    print("tokens:", decoded)
    print("ids:", encoded)
except ImportError:
    print("transformers not installed — pip install transformers")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from transformers import BertTokenizer`
- **Notable calls:** `BertTokenizer`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the IMDB sentiment dataset (small subset) and
> explores the distribution of review lengths.
> 
> **Why it matters:** review length affects how you tokenise, pad, and model.
> If 99% of reviews are < 500 tokens, you can safely truncate to 500. If
> you don't check, you may set `max_length=1000` and waste compute.
> 
> **What to watch for:** we use `datasets.load_dataset` (HuggingFace) because
> it streams well and handles the IMDB licensing cleanly. If you don't have
> the package, the code falls back to a synthetic corpus.


### Code cell #4
```python
try:
    from datasets import load_dataset
    ds = load_dataset("imdb", split="train[:500]")   # 500 samples for speed
    texts = [d["text"] for d in ds]
    labels = [d["label"] for d in ds]
except Exception as e:
    print(f"using synthetic corpus ({e})")
    texts = ["This movie was great!" * 5,
             "Awful, terrible, the worst." * 5,
             "I loved every minute." * 5] * 50
    labels = [1, 0, 1] * 50

import matplotlib.pyplot as plt
lengths = [len(t.split()) for t in texts]
plt.figure(figsize=(7, 3))
plt.hist(lengths, bins=30, color="steelblue", edgecolor="white")
plt.xlabel("review length (tokens)"); plt.ylabel("count")
plt.title("IMDB review length distribution")
plt.tight_layout(); plt.show()
print(f"mean length: {sum(lengths)/len(lengths):.0f}  median: {sorted(lengths)[len(lengths)//2]}")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from datasets import load_dataset`, `import matplotlib.pyplot as plt`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the top-20 most common tokens in the IMDB subset
> after stop-word removal. The shape is the classic Zipf distribution — a
> few tokens dominate, the long tail is huge.
> 
> **Why it matters:** Zipf's law ($f \propto 1/r$) is the universal shape of
> natural-language token frequencies. Models that ignore this (e.g. uniform
> regularisation across the vocabulary) waste capacity on rare tokens.
> 
> **What to watch for:** the y-axis is log-scaled. Zipf's law appears as a
> near-straight line on log-log axes.


### Code cell #5
```python
from collections import Counter
import numpy as np

all_tokens = []
for t in texts[:500]:
    all_tokens.extend(t.lower().split())
counter = Counter(all_tokens)
top20 = counter.most_common(20)

fig, ax = plt.subplots(figsize=(10, 4))
words, counts = zip(*top20)
ax.bar(words, counts, color="steelblue", edgecolor="white")
ax.set_xticklabels(words, rotation=45, ha="right")
ax.set_yscale("log")
ax.set_title("Top-20 tokens (Zipf's law on log-y)")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from collections import Counter`
- **Key imports:** `from collections import Counter`, `import numpy as np`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For text preprocessing, evaluate by:
> 
> - **Vocabulary size** — too small loses info, too large slows training.
> - **OOV rate** — fraction of tokens mapped to `<unk>`. Aim for <1%.
> - **Downstream model performance** — the only metric that ultimately matters.

> **What this does:** Computes the OOV rate on a held-out text snippet using
> our scratch `Vocab`. Anything in the test text that wasn't in the training
> vocabulary becomes `<unk>`.
> 
> **Why it matters:** if your OOV rate is high, your model can't generalise to
> new vocabulary. Sub-word tokenisers (BPE, WordPiece) reduce this dramatically.
> 
> **What to watch for:** the OOV rate here is artificially high because we
> trained the vocab on a tiny corpus. With a 10M-token corpus, OOV would
> be <1%.


### Code cell #6
```python
test_text = "The cat and the dog are friends forever."
test_tokens = tokenize(clean_text(test_text))
encoded = vocab.encode(test_tokens)
oov = sum(1 for i in encoded if i == 1)
print(f"test tokens: {test_tokens}")
print(f"encoded   : {encoded}")
print(f"OOV count : {oov} / {len(test_tokens)} = {oov/len(test_tokens):.1%}")
```

**What this code does:**

- **Headline:** `test_text = "The cat and the dog are friends forever."`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a stemmer (Porter or Snowball) and check the new vocabulary size.
> 2. Replace whitespace tokenisation with regex word tokenisation
>   (`re.findall(r"\b\w+\b", text)`).
> 3. Add bigrams to the bag-of-words vector (`ngram_range=(1, 2)`).
> 4. Use TF-IDF instead of raw counts (`TfidfVectorizer`).

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a Byte-Pair-Encoding (BPE) tokeniser from scratch.
> 2. Implement TF-IDF from scratch and verify against `TfidfVectorizer`.
> 3. Build a custom stop-word list for your specific domain.
> 4. Implement sub-word tokenisation for non-English text (e.g. Chinese).
> 5. Implement a sentence-splitter that handles abbreviations ("Dr. Smith").

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between stemming and lemmatisation?
> 2. What is a sub-word tokeniser, and why do modern models use it?
> 3. What is Zipf's law, and what does it imply for vocabulary design?
> 4. What is the OOV problem, and how do sub-words solve it?
> 5. Why is bag-of-words called *bag*?
> 6. What is the difference between `CountVectorizer` and `TfidfVectorizer`?
> 7. How would you handle very short texts (e.g. tweets) for tokenisation?
> 8. What is BPE, and what does `##` mean in BERT tokenisation?
