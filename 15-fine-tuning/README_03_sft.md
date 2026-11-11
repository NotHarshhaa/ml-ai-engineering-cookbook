# Code Walkthrough — `03_sft.ipynb`

> Module: **15-fine-tuning** | Notebook: **03_sft.ipynb**
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

> # 03 — SFT (Supervised Fine-Tuning)
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
> ## 1. Concept Overview — SFT — supervised fine-tuning

> SFT is the bridge from a base LLM (which predicts the next token) to an
> assistant LLM (which follows instructions). You fine-tune on (prompt,
> completion) pairs with the standard language-modelling loss — but only on
> the completion tokens.
> 
> **Beginner track:** learn the chat template, the loss-masking pattern,
> and the `SFTTrainer` API.
> **Advanced track:** also learn packing (concatenating multiple sequences),
> flash-attention, and gradient accumulation.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> SFT loss is cross-entropy on completion tokens only:
> $$L = -\sum_{t \in \text{completion}} \log P(x_t \mid x_{<t}; \theta)$$
> 
> Prompt tokens are *not* in the loss — they're context. This is called
> *completion-only loss* or *instruction loss masking*. The trick: build a
> labels tensor where prompt tokens are -100 (PyTorch's ignore_index).
> 
> Without masking, the model spends capacity learning to *predict* the
> prompt; with masking, all capacity goes to *completing* it.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements completion-only loss masking from scratch.
> We build a labels tensor from a (prompt, completion) pair, with -100 on
> prompt positions.
> 
> **Why it matters:** this is the one-line difference between SFT and plain
> language modelling. Get it wrong and the model learns to predict the
> instruction instead of following it.
> 
> **What to watch for:** in real code, the labels are token IDs for the
> completion and -100 for the prompt. PyTorch's `CrossEntropyLoss`
> automatically ignores -100.


### Code cell #1
```python
import torch
import torch.nn.functional as F

# Mock token IDs.
prompt_ids = [101, 102, 103, 104]   # tokens of the prompt
completion_ids = [201, 202, 203, 1024]  # tokens of the completion (last is EOS)

input_ids = torch.tensor([prompt_ids + completion_ids])

# Build labels: -100 for prompt, real IDs for completion.
labels = torch.full_like(input_ids, -100)
labels[:, len(prompt_ids):] = torch.tensor([completion_ids])

print("input_ids:", input_ids.tolist())
print("labels   :", labels.tolist())
print("(labels show -100 on prompt, real IDs on completion)")

# Mock logits: (batch, seq, vocab).
torch.manual_seed(0)
vocab_size = 1025
logits = torch.randn(1, len(prompt_ids) + len(completion_ids), vocab_size)

# Cross-entropy loss with ignore_index=-100.
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
print(f"\nSFT loss (only completion tokens): {loss.item():.4f}")
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`, `import torch.nn.functional as F`
- **Notable calls:** `cross_entropy`

> **Takeaway:** SFT = LM loss + labels with -100 on prompt. The masking is
> the entire difference from base LM training. Get it right and the model
> learns to *complete* instructions, not predict them.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `trl.SFTTrainer` — the standard SFT training
> library. We show the pattern for training on a chat dataset with completion
> loss masking.
> 
> **Why it matters:** `SFTTrainer` handles tokenisation, chat templating,
> loss masking, packing, and gradient accumulation. Doing it manually is
> error-prone — use the library.
> 
> **What to watch for:** `dataset_text_field=field contains the formatted chat. Use `formatting_func` for custom
> templates.


### Code cell #2
```python
try:
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer
    from datasets import Dataset

    print("trl installed; pattern would be:")
    print("""  tokenizer = AutoTokenizer.from_pretrained(...)
  dataset = Dataset.from_list(sft_data)
  config = SFTConfig(output_dir='./out', num_train_epochs=3,
                     per_device_train_batch_size=4, packing=True)
  trainer = SFTTrainer(
                     model=model,
                     args=config,
                     train_dataset=dataset,
                     tokenizer=tokenizer,
                     formatting_func=format_chat)
  trainer.train()""")
except ImportError as e:
    print(f"trl not installed ({e}) — pip install trl")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from trl import SFTTrainer`, `from transformers import AutoTokenizer`, `from datasets import Dataset`
- **Notable calls:** `Dataset`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a slightly larger SFT dataset (100 examples) by
> templating the 10 base examples with variations. Useful for demonstrating
> the data format that real SFT uses.
> 
> **Why it matters:** SFT datasets are the *product* — the model is just the
> engine. A clean, well-formatted dataset matters more than the choice of
> model.
> 
> **What to watch for:** real SFT datasets are 1k-100k examples. <1k risks
> overfitting; >100k takes too long to label.


### Code cell #3
```python
import random
random.seed(0)

# Templates for variation.
templates = [
    lambda p, c: {"prompt": p, "completion": c},
    lambda p, c: {"prompt": f"Please answer: {p}", "completion": c},
    lambda p, c: {"prompt": f"Question: {p}\nAnswer:", "completion": f" {c}"},
    lambda p, c: {"prompt": f"Q: {p}", "completion": f"A: {c}"},
]

expanded_sft = []
for ex in sft_data:
    for tmpl in templates:
        expanded_sft.append(tmpl(ex["prompt"], ex["completion"]))
print(f"expanded SFT dataset: {len(expanded_sft)} examples")
for ex in expanded_sft[:4]:
    print(f"  prompt: {ex['prompt']!r}")
    print(f"  completion: {ex['completion']!r}")
```

**What this code does:**

- **Headline:** `import random`
- **Key imports:** `import random`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the loss curve of a mock SFT run. Real SFT loss
> curves drop quickly in the first 100 steps (the model learns the chat
> template), then plateau.
> 
> **Why it matters:** the loss curve tells you whether SFT is working. If it
> drops sharply, the model is learning the template. If it stays flat, your
> data or learning rate is wrong.
> 
> **What to watch for:** the initial loss reflects the model's confusion
> about the chat format. After 100 steps it should be much lower.


### Code cell #4
```python
import matplotlib.pyplot as plt
import numpy as np

# Mock SFT loss curve: sharp drop, then slow decay.
steps = np.arange(0, 500)
loss = 3.0 * np.exp(-steps / 50) + 1.5 + 0.1 * np.random.default_rng(0).standard_normal(500)

plt.figure(figsize=(9, 4))
plt.plot(steps, loss, color="steelblue", lw=1)
plt.xlabel("training step"); plt.ylabel("SFT loss")
plt.title("Typical SFT loss curve: rapid initial drop, then slow decay")
plt.yscale("log")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import numpy as np`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> SFT is evaluated by:
> 
> - **Completion loss** — training + validation, should decrease smoothly.
> - **Instruction-following accuracy** — on a held-out set of instructions.
> - **Chat format compliance** — does the model produce well-formed responses?
> - **Human evaluation** — for assistant quality, the ultimate metric.

> **What this does:** A simple post-SFT eval: feed the model a held-out
> instruction, generate a response, and check it contains the gold answer.
> 
> **Why it matters:** SFT changes the model's behaviour — it's no longer
> just predicting text. You need a behaviour eval, not just a perplexity eval.
> 
> **What to watch for:** for real eval, generate 5-10 responses per
> instruction (temperature=0.7) and pick the majority answer. Single-sample
> eval is too noisy.


### Code cell #5
```python
# Held-out eval set (different from the training set).
sft_eval = [
    {"prompt": "What is 5 + 5?",          "gold": "10"},
    {"prompt": "Capital of Japan?",        "gold": "Tokyo"},
    {"prompt": "Translate 'good' to French.", "gold": "bon"},
    {"prompt": "Reverse 'hello'.",        "gold": "olleh"},
]

# Mock model: pretend it learned the format.
def mock_sft_model(prompt):
    # In real code: model.generate(tokenizer(prompt), ...)
    return f"The answer is: {sft_eval[0]['gold']}" if "5 + 5" in prompt else "I don't know."

correct = 0
for qa in sft_eval:
    pred = mock_sft_model(qa["prompt"])
    ok = qa["gold"].lower() in pred.lower()
    correct += int(ok)
    print(f"  Q: {qa['prompt'][:30]:32s} gold: {qa['gold']:10s} ok: {ok}")
print(f"\nSFT eval accuracy: {correct}/{len(sft_eval)}")
```

**What this code does:**

- **Headline:** `sft_eval = [`
- **Defines:** def `mock_sft_model`
- **Notable calls:** `model.generate`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Train with vs without completion loss masking — does the model follow instructions?
> 2. Use `packing=True` in `SFTConfig` — does it speed up training?
> 3. Add a system message to the chat template — does it help with persona?
> 4. Train with LoRA + SFT (QLoRA) on a 1k-example dataset.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement SFT from scratch (without `SFTTrainer`) — use `transformers.Trainer` with custom collator.
> 2. Build a chat-template formatter that handles multi-turn conversations.
> 3. Implement an instruction-following eval (use LLM-as-judge).
> 4. Add gradient accumulation to simulate batch size 64 with 4-GPU effective batch.
> 5. Implement a packing-aware attention mask for sequence packing.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between pretraining and SFT?
> 2. Why do we mask the prompt tokens out of the SFT loss?
> 3. What is a chat template, and why does it matter?
> 4. What is sequence packing, and when does it help?
> 5. How would you evaluate an SFT model?
> 6. What is gradient accumulation, and why use it?
> 7. How do you handle multi-turn conversations in SFT?
> 8. What are common failure modes of SFT?
