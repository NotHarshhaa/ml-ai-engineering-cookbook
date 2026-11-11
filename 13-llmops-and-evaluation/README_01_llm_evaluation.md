# Code Walkthrough — `01_llm_evaluation.ipynb`

> Module: **13-llmops-and-evaluation** | Notebook: **01_llm_evaluation.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 29
- **Markdown cells:** 24
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

> # 01 — LLM Evaluation
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
> ## 1. Concept Overview — LLM evaluation

> Evaluating LLMs is harder than evaluating classifiers. There's no single
> ground-truth label; outputs are open-ended; "better" is often subjective.
> This notebook covers the standard approaches: exact-match for closed
> tasks, LLM-as-judge for open tasks, and benchmark suites like MMLU.
> 
> **Beginner track:** learn the three eval patterns — exact match, semantic
> match, LLM-as-judge. They cover 90% of LLM eval.
> **Advanced track:** also learn the pitfalls of LLM-as-judge (position bias,
> verbosity bias) and how to mitigate them.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Exact match: $\text{EM}(\hat{y}, y) = \mathbb{1}[\hat{y} = y]$.
> 
> Semantic match: $\text{sim}(\hat{y}, y)$ — typically via embedding cosine or
> an LLM that judges equivalence.
> 
> LLM-as-judge: ask a stronger LLM (e.g. GPT-4) to rate the candidate output
> on a scale (1-5) or pick between A/B. Score = average over the eval set.
> 
> Perplexity: $\exp(-\frac{1}{T} \sum_t \log P(x_t \mid x_{<t}))$ — for
> language modelling tasks.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements three LLM evaluators in pure Python:
> exact-match, BLEU (a classic MT metric), and a mock LLM-as-judge.
> 
> **Why it matters:** seeing these in 50 lines removes the magic of
> evaluation frameworks. Each is a simple function over (prediction, gold).
> 
> **What to watch for:** BLEU uses n-gram overlap with brevity penalty. It's
> the standard translation metric but has known flaws — a high BLEU doesn't
> guarantee a good translation.


### Code cell #1
```python
import math
from collections import Counter

def exact_match(pred, gold):
    """1 if exact string match, else 0."""
    return int(pred.strip().lower() == gold.strip().lower())

def bleu(pred, gold, max_n=4):
    """BLEU-n: n-gram precision with brevity penalty."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens: return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1))
        gold_ngrams = Counter(tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens)-n+1))
        overlap = sum(min(c, gold_ngrams.get(g, 0)) for g, c in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append(overlap / max(1, total))
    # Brevity penalty.
    bp = 1.0 if len(pred_tokens) > len(gold_tokens) else math.exp(1 - len(gold_tokens)/max(1, len(pred_tokens)))
    if min(precisions) == 0: return 0.0
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)

def llm_as_judge_mock(pred, gold, question):
    """Mock LLM judge: returns 5 if pred contains gold's key words, else 3."""
    gold_words = set(gold.lower().split())
    pred_words = set(pred.lower().split())
    overlap = len(gold_words & pred_words) / max(1, len(gold_words))
    return 5 if overlap > 0.5 else 3 if overlap > 0.2 else 1

print("exact match:", exact_match("Paris", "paris"))
print("BLEU:", bleu("the cat sat on the mat", "the cat sat on a mat"))
print("LLM judge:", llm_as_judge_mock("Paris is the capital of France", "Paris", "capital of France?"))
```

**What this code does:**

- **Headline:** `import math`
- **Defines:** def `exact_match` — 1 if exact string match, else 0.; def `bleu` — BLEU-n: n-gram precision with brevity penalty.; def `llm_as_judge_mock` — Mock LLM judge: returns 5 if pred contains gold's key words, else 3.
- **Key imports:** `import math`, `from collections import Counter`

> **Takeaway:** LLM evaluation = scoring functions over (prediction, gold).
> Exact match is brittle, BLEU is partial, LLM-as-judge is flexible but
> expensive. Pick the metric that matches your task.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `rouge_score` and `nltk.translate.bleu_score` —
> the production-grade implementations of the classic NLP metrics. We
> compare scratch vs library on the same example.
> 
> **Why it matters:** the scratch implementations are for learning; for
> real evaluation, use these libraries. They handle edge cases (smoothing,
> tokenisation) that the scratch versions skip.
> 
> **What to watch for:** BLEU has multiple smoothing methods; the default
> (Method 0) returns 0 if any n-gram precision is 0, which is too harsh for
> short outputs. Use Method 1 or 2 for short-text eval.


### Code cell #2
```python
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    pred = "the cat sat on the mat".split()
    gold = "the cat sat on a mat".split()
    smoothie = SmoothingFunction().method1
    bleu_lib = sentence_bleu([gold], pred, smoothing_function=smoothie)
    print(f"library BLEU: {bleu_lib:.3f}")
    print(f"scratch BLEU: {bleu('the cat sat on the mat', 'the cat sat on a mat'):.3f}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = scorer.score("the cat sat on a mat", "the cat sat on the mat")
    print(f"ROUGE-1: {scores['rouge1']}")
    print(f"ROUGE-L: {scores['rougeL']}")
except ImportError as e:
    print(f"install nltk and rouge_score: pip install nltk rouge_score ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from nltk.translate.bleu_score import sentence_bleu`, `from rouge_score import rouge_scorer`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a tiny LLM eval set: 5 (question, gold answer)
> pairs from common benchmarks. We'll use it to demonstrate each metric.
> 
> **Why it matters:** a good eval set is *representative* of your task. Our
> set is mixed (factual, math, code, summarisation) so different metrics
> shine on different items.
> 
> **What to watch for:** the gold answers here are short — perfect for
> exact-match. For long-form generation, exact match is useless; use
> ROUGE or LLM-as-judge.


### Code cell #3
```python
llm_eval = [
    {"q": "What is the capital of France?", "gold": "Paris",         "type": "factual"},
    {"q": "What is 12 * 7?",                 "gold": "84",           "type": "math"},
    {"q": "Reverse the string 'abc'.",       "gold": "cba",          "type": "code"},
    {"q": "Summarise: The cat sat on the mat.", "gold": "A cat sat on a mat.", "type": "summary"},
    {"q": "Translate 'hello' to French.",     "gold": "bonjour",      "type": "translation"},
]
print(f"eval set size: {len(llm_eval)}")
for qa in llm_eval:
    print(f"  Q: {qa['q']}  gold: {qa['gold']}  type: {qa['type']}")
```

**What this code does:**

- **Headline:** `llm_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the score distribution of three LLM outputs
> (perfect, partial, wrong) across our three metrics. The point: each
> metric tells a different story.
> 
> **Why it matters:** picking the right metric is *the* decision in LLM
> evaluation. Plotting the same set under different metrics makes the
> trade-off visible.
> 
> **What to watch for:** exact match is binary (0 or 1), BLEU is continuous
> but harsh, LLM-judge is continuous and lenient. Choose based on the cost
> of false positives vs false negatives in your application.


### Code cell #4
```python
import matplotlib.pyplot as plt
import numpy as np

# Three mock predictions for the same gold answer.
gold = "the cat sat on the mat"
preds = [
    ("perfect",  "the cat sat on the mat"),
    ("partial",  "the cat sat on a mat"),
    ("wrong",    "the dog ran in the park"),
]

metrics = []
for name, p in preds:
    metrics.append({
        "name": name,
        "exact": exact_match(p, gold),
        "bleu":  bleu(p, gold),
        "judge": llm_as_judge_mock(p, gold, "") / 5.0,  # scale to 0-1
    })

names = [m["name"] for m in metrics]
exact_vals = [m["exact"] for m in metrics]
bleu_vals  = [m["bleu"]  for m in metrics]
judge_vals = [m["judge"] for m in metrics]

x = np.arange(len(names))
width = 0.25
plt.figure(figsize=(8, 4))
plt.bar(x - width, exact_vals, width, label="exact match", color="#3498db")
plt.bar(x,         bleu_vals,  width, label="BLEU",         color="#27ae60")
plt.bar(x + width, judge_vals, width, label="LLM-judge",   color="#e74c3c")
plt.xticks(x, names)
plt.ylabel("score (0-1)")
plt.title("Three LLM outputs scored by three metrics")
plt.legend()
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import numpy as np`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> LLM evaluation has three orthogonal axes:
> 
> - **Accuracy** — does the output answer the question correctly?
> - **Safety** — is the output free of harmful content (covered in notebook 2)?
> - **Helpfulness** — does the output actually help the user?
> 
> Always evaluate on a *held-out* set the model has not seen, and report
> multiple metrics — never just one.

> **What this does:** Runs all three metrics on a mock LLM that produces
> correct answers for factual questions and wrong answers for everything
> else. We report the per-question scores and aggregate.
> 
> **Why it matters:** this is the *minimum* eval script you'd write before
> deploying any LLM. Per-question scores show you *where* the model fails;
> aggregate scores give the headline number.
> 
> **What to watch for:** we mock the LLM (no real model). In production,
> you'd replace `mock_llm` with a call to your model's `generate` function.


### Code cell #5
```python
# Mock LLM that gets factual questions right and others wrong.
def mock_llm(q):
    q_lower = q.lower()
    if "capital of france" in q_lower: return "Paris"
    if "12 * 7" in q_lower: return "84"
    if "abc" in q_lower and "reverse" in q_lower: return "cba"
    if "summarise" in q_lower or "summarize" in q_lower: return "A cat sits."
    if "translate" in q_lower and "hello" in q_lower: return "bonjour"
    return "I don't know."

scores = []
for qa in llm_eval:
    pred = mock_llm(qa["q"])
    em = exact_match(pred, qa["gold"])
    bl = bleu(pred, qa["gold"])
    jd = llm_as_judge_mock(pred, qa["gold"], qa["q"])
    scores.append({"q": qa["q"], "pred": pred, "em": em, "bleu": bl, "judge": jd})
    print(f"  Q: {qa['q'][:40]:42s} pred: {pred[:20]:22s} EM={em} BLEU={bl:.2f} judge={jd}")

print(f"\nmean exact match: {np.mean([s['em'] for s in scores]):.2f}")
print(f"mean BLEU       : {np.mean([s['bleu'] for s in scores]):.2f}")
print(f"mean LLM judge  : {np.mean([s['judge'] for s in scores]):.2f}/5")
```

**What this code does:**

- **Headline:** `def mock_llm(q):`
- **Defines:** def `mock_llm`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a real LLM (e.g. GPT-2 from `transformers`) and re-run the eval.
> 2. Compare exact-match vs ROUGE-L on a summarisation task — which is more fair?
> 3. Use a real LLM as judge (e.g. `gpt-4o-mini`) and compare to the mock.
> 4. Add a *bias check*: ask the same question phrased differently and check for consistency.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement an LLM-as-judge with position-balanced scoring (swap A/B and average).
> 2. Build an MMLU-style multiple-choice eval set with 100 questions.
> 3. Implement a confusion matrix for open-ended LLM outputs (by topic).
> 4. Add a *verbosity bias* check: penalise outputs > 2x the gold length.
> 5. Implement a regression test: re-evaluate the model after every code change.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why is LLM evaluation harder than classifier evaluation?
> 2. What is BLEU, and when is it a poor metric?
> 3. What is LLM-as-judge, and what are its known biases?
> 4. How do you mitigate position bias in LLM-as-judge?
> 5. What is ROUGE-L, and when would you use it over BLEU?
> 6. How would you evaluate a multi-turn conversation model?
> 7. What is perplexity, and why doesn't it always correlate with quality?
> 8. How would you set up a regression test for an LLM application?
