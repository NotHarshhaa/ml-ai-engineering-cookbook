# Code Walkthrough — `03_bert_mlm.ipynb`

> Module: **09-transformers-and-llms** | Notebook: **03_bert_mlm.ipynb**
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

> # 03 — BERT & Masked Language Modeling
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
> ## 1. Concept Overview — BERT and masked language modelling

> BERT (Bidirectional Encoder Representations from Transformers, Devlin 2018)
> is the canonical *encoder* transformer. It's pretrained by predicting masked
> tokens in random positions (the *MLM* objective). After pretraining, you
> fine-tune it on a downstream task by adding a classification head.
> 
> **Beginner track:** understand the MLM objective and why BERT is
> *bidirectional* (it sees the whole sequence at once).
> **Advanced track:** also learn BERT's training tricks (next-sentence
> prediction, WordPiece, segment embeddings) and how modern variants
> (RoBERTa, DeBERTa, ALBERT) improve on it.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> **Masked LM objective**: select 15% of tokens at random. Of those:
> - 80% are replaced with `[MASK]`.
> - 10% are replaced with a random token.
> - 10% are left unchanged.
> 
> Loss = cross-entropy on the masked positions only.
> 
> Fine-tuning: prepend a `[CLS]` token, take its final hidden state, feed to
> a linear classifier for the downstream task.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements the BERT-style masked-LM pretraining step in
> pure NumPy: select 15% of tokens, apply the 80/10/10 rule, then compute
> cross-entropy loss on the masked positions.
> 
> **Why it matters:** the *80/10/10 trick* is what makes BERT robust at
> fine-tuning time, when there's no `[MASK]` token in the input. If you only
> used `[MASK]`, the model would learn nothing about the original tokens.
> 
> **What to watch for:** the loss is averaged *only* over masked positions.
> Computing it over all positions would waste training signal on tokens the
> model already saw.


### Code cell #1
```python
import numpy as np

def mask_tokens(input_ids, vocab_size, mask_token_id, mask_prob=0.15, seed=0):
    """Apply BERT-style 80/10/10 masking to a sequence."""
    rng = np.random.default_rng(seed)
    masked_labels = input_ids.copy()
    # Pick which positions to mask (with probability mask_prob).
    mask = rng.uniform(size=len(input_ids)) < mask_prob
    masked_input = input_ids.copy()
    for i in np.where(mask)[0]:
        r = rng.uniform()
        if r < 0.8:                  # 80% -> [MASK]
            masked_input[i] = mask_token_id
        elif r < 0.9:                # 10% -> random token
            masked_input[i] = rng.integers(0, vocab_size)
        # else: 10% -> leave unchanged
    return masked_input, masked_labels, mask

# Toy vocab of 10 tokens. Let token 0 be [MASK].
input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1])
mi, ml, mask = mask_tokens(input_ids, vocab_size=10, mask_token_id=0, seed=0)
print("input      :", input_ids)
print("masked     :", mi)
print("labels     :", ml)
print("mask?      :", mask)
print("loss is computed only on positions where mask=True:", np.where(mask)[0])
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `mask_tokens` — Apply BERT-style 80/10/10 masking to a sequence.
- **Key imports:** `import numpy as np`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Loads a pretrained BERT-base model from HuggingFace
> and runs it on a sample sentence to extract token-level embeddings. If
> `transformers` isn't installed, we skip the demo gracefully.
> 
> **Why it matters:** this is the *production* way to use BERT — you rarely
> train one from scratch. Loading and using pretrained BERT is the foundation
> of almost all modern NLP.
> 
> **What to watch for:** `bert-base-uncased` means lowercase all text and
> merge accents. Use `bert-base-cased` if case matters for your task.


### Code cell #2
```python
try:
    from transformers import BertTokenizer, BertModel
    import torch
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    text = "The cat sat on the mat because it was warm."
    inputs = tokenizer(text, return_tensors="pt")
    print("token IDs:", inputs["input_ids"].tolist())
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (batch, seq_len, hidden_dim)
    print("embeddings shape:", outputs.last_hidden_state.shape)
    print("[CLS] embedding (first 5 dims):", outputs.last_hidden_state[0, 0, :5].round(3).tolist())
except ImportError:
    print("transformers not installed — pip install transformers")
except Exception as e:
    print(f"skipped BERT demo (no network?): {e}")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from transformers import BertTokenizer`, `import torch`
- **Notable calls:** `BertTokenizer`, `BertModel`, `model.eval`

> **What this does:** Uses BERT for the *masked LM* task it was pretrained on.
> We mask one token, ask BERT to fill it in, and show the top-5 predictions.
> 
> **Why it matters:** this is the canonical "is BERT actually working?" test.
> If BERT's top prediction for "The cat sat on the [MASK]." is "mat",
> you've confirmed the model has language understanding.
> 
> **What to watch for:** we use `BertForMaskedLM` (with the LM head) instead
> of `BertModel` (encoder only). The LM head is what produces token-level
> logits.


### Code cell #3
```python
try:
    from transformers import BertTokenizer, BertForMaskedLM
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    mlm = BertForMaskedLM.from_pretrained("bert-base-uncased")
    mlm.eval()

    text = "The cat sat on the [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()

    with torch.no_grad():
        logits = mlm(**inputs).logits[0, mask_pos]   # (vocab,)
    top5 = logits.topk(5)
    print(f"text: {text}")
    print("top-5 BERT predictions for [MASK]:")
    for idx, score in zip(top5.indices, top5.values):
        print(f"  {tokenizer.decode([idx]):10s}  (logit={score:.2f})")
except ImportError:
    print("transformers not installed — pip install transformers")
except Exception as e:
    print(f"skipped MLM demo (no network?): {e}")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from transformers import BertTokenizer`
- **Notable calls:** `BertTokenizer`, `BertForMaskedLM`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the GLUE SST-2 sentiment dataset (a standard
> fine-tuning benchmark) and explores its class distribution. Falls back to
> a tiny synthetic dataset if `datasets` is unavailable.
> 
> **Why it matters:** SST-2 is *the* benchmark for sentence classification
> fine-tuning. BERT-base fine-tuned on it reaches ~92% accuracy; that's the
> standard reference.
> 
> **What to watch for:** SST-2 is binary (positive/negative). The full GLUE
> benchmark has 11 tasks with different formats — be aware of the variation.


### Code cell #4
```python
try:
    from datasets import load_dataset
    sst2 = load_dataset("glue", "sst2")
    train = sst2["train"]
    print("dataset size:", len(train))
    print("columns:", train.column_names)
    print("first example:", train[0])
    print("\nlabel distribution:")
    print({k: v for k, v in zip(*np.unique([d["label"] for d in train], return_counts=True))})
except Exception as e:
    print(f"using synthetic dataset ({e})")
    train = [{"sentence": "This movie was great!", "label": 1},
             {"sentence": "Terrible, awful, the worst.", "label": 0}] * 50
    print("synthetic train size:", len(train))
    print("first example:", train[0])
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from datasets import load_dataset`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the BERT [CLS] embedding for 4 sentences in 2-D
> (via PCA). Sentences about the same topic should cluster together.
> 
> **Why it matters:** this is the standard sanity check that BERT *learned*
> semantic representations. If positive and negative sentences cluster
> apart, you know the [CLS] token is meaningful.
> 
> **What to watch for:** with only 4 sentences, the PCA is unstable. In a
> real evaluation you'd plot 100+ sentences grouped by sentiment.


### Code cell #5
```python
import matplotlib.pyplot as plt
try:
    sentences = [
        "The cat sat on the mat.",
        "A dog rested on the rug.",
        "I love this great movie!",
        "This terrible film was awful.",
    ]
    embs = []
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=20)
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[0, 0]   # [CLS]
        embs.append(cls.numpy())
    embs = np.array(embs)
    from sklearn.decomposition import PCA
    Z = PCA(n_components=2).fit_transform(embs)
    plt.figure(figsize=(7, 6))
    for i, s in enumerate(sentences):
        plt.scatter(Z[i, 0], Z[i, 1], s=80)
        plt.annotate(s[:25], (Z[i, 0], Z[i, 1]), fontsize=9, xytext=(5, 5),
                     textcoords="offset points")
    plt.title("BERT [CLS] embeddings (PCA 2-D) for 4 sentences")
    plt.tight_layout(); plt.show()
except Exception as e:
    print(f"skipped BERT embedding plot ({e})")
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `from sklearn.decomposition import PCA`
- **Notable calls:** `PCA`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> BERT-style models are evaluated by:
> 
> - **MLM accuracy** during pretraining — how well does it fill in masks?
> - **Downstream task accuracy** after fine-tuning (SST-2, MNLI, SQuAD).
> - **GLUE / SuperGLUE / BIG-bench** — benchmark suites.

> **What this does:** A skeleton for fine-tuning BERT on SST-2. We show the
> exact API — `BertForSequenceClassification` + `Trainer` — without running
> the full training (which takes ~30 min on a GPU).
> 
> **Why it matters:** this is the standard pattern for fine-tuning any
> transformer in HuggingFace. The `Trainer` API handles batching, gradient
> accumulation, mixed precision, evaluation, and checkpointing.
> 
> **What to watch for:** `Trainer` is convenient but hides a lot. For research
> you may prefer a manual training loop (more control, easier to debug).


### Code cell #6
```python
try:
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    # Build a 2-class classification head on top of BERT.
    cls_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    print("model loaded with classification head; parameter count:",
          sum(p.numel() for p in cls_model.parameters()))
    # We would normally call Trainer.train() here, but skip the actual training
    # to avoid the GPU requirement.
    print("(training skipped in this notebook — try on Colab for full demo)")
except Exception as e:
    print(f"skipped fine-tuning demo ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from transformers import BertForSequenceClassification`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Mask 30% of tokens instead of 15% — does MLM accuracy drop?
> 2. Use `bert-large-uncased` and compare embeddings.
> 3. Replace BERT with `roberta-base` — does the MLM accuracy differ?
> 4. Fine-tune on SST-2 for 1 epoch and report test accuracy.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement BERT-style MLM pretraining from scratch in PyTorch.
> 2. Implement Next Sentence Prediction (NSP) — BERT's second pretraining task.
> 3. Implement a custom fine-tuning loop (without `Trainer`) for SST-2.
> 4. Add LoRA adapters to a pretrained BERT for parameter-efficient fine-tuning.
> 5. Distil a BERT-base model into a 4-layer student (knowledge distillation).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the MLM objective, and why does it use the 80/10/10 trick?
> 2. Why is BERT *bidirectional* while GPT is not?
> 3. What is the [CLS] token, and why is it used for classification?
> 4. What is the difference between BERT, RoBERTa, and DeBERTa?
> 5. How is fine-tuning different from pretraining?
> 6. What is the role of segment embeddings in BERT?
> 7. Why does BERT cap sequences at 512 tokens?
> 8. What is LoRA, and what problem does it solve?
