# Code Walkthrough — `04_dpo.ipynb`

> Module: **15-fine-tuning** | Notebook: **04_dpo.ipynb**
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

> # 04 — DPO (Direct Preference Optimization)
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
> ## 1. Concept Overview — DPO — direct preference optimization

> After SFT, the model can follow instructions — but it doesn't know which
> responses humans *prefer*. DPO is a simple, RLHF-free method: fine-tune
> on (prompt, chosen, rejected) preference pairs, where `chosen` was
> preferred by a human or a stronger model.
> 
> **Beginner track:** understand the DPO loss formula and the preference
> dataset format.
> **Advanced track:** also learn IPO, KTO, and the relationship between DPO
> and RLHF/PPO.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> DPO loss (simplified):
> $$L_\text{DPO} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)}
> - \beta \log\frac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)}\right)$$
> 
> where $y_w$ is the chosen (winning) response, $y_l$ is the rejected (losing)
> response, $\pi_\theta$ is the policy being trained, $\pi_\text{ref}$ is the
> reference (frozen) policy, and $\beta$ is a temperature (typically 0.1).
> 
> Intuition: increase the log-likelihood ratio of chosen vs rejected,
> relative to the reference. The sigmoid + log turns this into a binary
> classification loss.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements the DPO loss in pure PyTorch. We use a
> mock policy and reference policy, both producing token log-probabilities,
> and compute the loss on a single (chosen, rejected) pair.
> 
> **Why it matters:** DPO looks intimidating until you see it's just a
> logistic loss on log-ratio differences. ~10 lines of real math.
> 
> **What to watch for:** `log_probs` must be summed over the response
> tokens (mean over tokens, then sum). Padding tokens should be masked out.


### Code cell #1
```python
import torch
import torch.nn.functional as F

def per_token_log_probs(logits, labels):
    """Log probability of each label under the logits (per position)."""
    log_p = F.log_softmax(logits, dim=-1)
    return log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps,    ref_rejected_logps, beta=0.1):
    """Standard DPO loss. Each *_logps is shape (batch,)."""
    # Log-ratios: log pi(y|x) - log pi_ref(y|x).
    chosen_logratios   = policy_chosen_logps   - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    # The DPO loss: -log sigmoid(beta * (chosen - rejected)).
    logits = beta * (chosen_logratios - rejected_logratios)
    return -F.logsigmoid(logits).mean()

# Mock: 1 batch with 4-token responses.
torch.manual_seed(0)
vocab = 100
# Policy and reference logits for chosen and rejected responses.
policy_chosen_logits  = torch.randn(1, 4, vocab)
policy_rejected_logits = torch.randn(1, 4, vocab)
ref_chosen_logits     = torch.randn(1, 4, vocab)
ref_rejected_logits   = torch.randn(1, 4, vocab)

# Mock labels (the "true" tokens of chosen and rejected responses).
chosen_labels   = torch.randint(0, vocab, (1, 4))
rejected_labels = torch.randint(0, vocab, (1, 4))

# Compute per-token log-probs, then sum over tokens.
pc = per_token_log_probs(policy_chosen_logits,  chosen_labels).sum(-1)
pr = per_token_log_probs(policy_rejected_logits, rejected_labels).sum(-1)
rc = per_token_log_probs(ref_chosen_logits,     chosen_labels).sum(-1)
rr = per_token_log_probs(ref_rejected_logits,   rejected_labels).sum(-1)

loss = dpo_loss(pc, pr, rc, rr, beta=0.1)
print(f"DPO loss: {loss.item():.4f}")
print(f"(lower = policy aligns better with preferences)")
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** def `per_token_log_probs` — Log probability of each label under the logits (per position).; def `dpo_loss` — Standard DPO loss. Each *_logps is shape (batch,).
- **Key imports:** `import torch`, `import torch.nn.functional as F`
- **Notable calls:** `softmax`

> **Takeaway:** DPO = logistic loss on log-ratio differences. The reference
> model anchors the policy to prevent drift. No RL, no reward model — just
> supervised learning on preferences.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `trl.DPOTrainer` — the standard DPO training
> library. We show the pattern for training on a preference dataset.
> 
> **Why it matters:** `DPOTrainer` handles reference model management,
> log-prob computation, and the loss. Doing it manually is error-prone.
> 
> **What to watch for:** `beta=0.1` is the default; higher beta = more
> aggressive divergence from the reference, lower = more conservative. Most
> use cases use 0.1.


### Code cell #2
```python
try:
    from trl import DPOTrainer, DPOConfig

    print("trl installed; DPO pattern would be:")
    print("  dpo_config = DPOConfig(")
    print("      beta=0.1,")
    print("      output_dir='./dpo_out',")
    print("      per_device_train_batch_size=4,")
    print("      num_train_epochs=1,")
    print("  )")
    print("  trainer = DPOTrainer(")
    print("      model=model,")
    print("      ref_model=None,  # uses the same model as reference")
    print("      args=dpo_config,")
    print("      train_dataset=preference_dataset,")
    print("      tokenizer=tokenizer,")
    print("  )")
    print("  trainer.train()")
except ImportError as e:
    print(f"trl not installed ({e})")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from trl import DPOTrainer`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small preference dataset: each example has a
> prompt, a chosen (preferred) response, and a rejected response. This is
> the input format for DPO.
> 
> **Why it matters:** preference data is the *bottleneck* of DPO. Real
> datasets come from human raters (expensive) or a stronger LLM judge
> (cheaper, but biased).
> 
> **What to watch for:** the chosen and rejected responses should be *the
> same length* on average. Length bias can leak into DPO — the model learns
> to prefer shorter or longer responses regardless of quality.


### Code cell #3
```python
preference_data = [
    {"prompt": "What is the capital of France?",
     "chosen": "Paris is the capital of France.",
     "rejected": "I don't know."},
    {"prompt": "Explain gravity in one sentence.",
     "chosen": "Gravity is the force that pulls objects toward each other based on their mass.",
     "rejected": "Gravity is a thing that exists."},
    {"prompt": "Write a haiku about autumn.",
     "chosen": "Leaves drift to the ground,\nGolden red beneath my feet,\nCool breeze whispers low.",
     "rejected": "Autumn is nice I guess."},
    {"prompt": "What is 2 + 2?",
     "chosen": "2 + 2 = 4.",
     "rejected": "2 + 2 = 5."},
]
print(f"preference dataset size: {len(preference_data)}")
for ex in preference_data[:2]:
    print(f"  prompt: {ex['prompt']}")
    print(f"  chosen:  {ex['chosen'][:50]}...")
    print(f"  rejected: {ex['rejected'][:50]}...")
```

**What this code does:**

- **Headline:** `preference_data = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the DPO loss landscape: as the policy's
> log-ratio for chosen increases (vs rejected), the loss decreases. The
> sigmoid shape is the binary classification core of DPO.
> 
> **Why it matters:** seeing the loss landscape helps you understand *what*
> DPO optimises. It's a soft classification of (chosen, rejected) pairs by
> their log-prob ratios.
> 
> **What to watch for:** the loss is flat when the policy already prefers
> chosen; steep when it's wrong. This is exactly the binary classification
> shape.


### Code cell #4
```python
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# DPO loss as a function of the log-ratio difference (chosen - rejected).
diffs = torch.linspace(-5, 5, 100)
for beta in [0.05, 0.1, 0.3]:
    losses = -F.logsigmoid(beta * diffs)
    plt.plot(diffs.numpy(), losses.numpy(), label=f"beta={beta}")
plt.xlabel("log-ratio difference (chosen - rejected)")
plt.ylabel("DPO loss")
plt.title("DPO loss as a function of log-ratio difference")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import torch`, `import torch.nn.functional as F`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> DPO is evaluated by:
> 
> - **Reward accuracy** — does the policy assign higher log-prob to chosen vs rejected?
> - **KL from reference** — small KL = conservative; large KL = aggressive.
> - **Downstream task quality** — human eval, LLM-as-judge, or benchmark accuracy.
> - **Length and bias checks** — DPO can introduce length and verbosity bias.

> **What this does:** Computes the reward accuracy of our mock policy on
> the preference dataset: how often does the policy assign a higher
> log-prob to the chosen vs the rejected response?
> 
> **Why it matters:** reward accuracy is the *direct* measure of DPO success.
> If it's >90% on the eval set, the policy has internalised the preferences.
> 
> **What to watch for:** real DPO eval uses a held-out preference set (not
> the training set) and a real LLM (not a mock). The principle is the same.


### Code cell #5
```python
# Mock: a policy that gives chosen a +0.5 log-prob boost vs rejected.
import torch
import torch.nn.functional as F

correct = 0
for ex in preference_data:
    # Mock policy log-probs: chosen gets +0.5, rejected gets -0.5.
    chosen_lp = 0.5 + 0.1 * torch.randn(1).item()
    rejected_lp = -0.5 + 0.1 * torch.randn(1).item()
    if chosen_lp > rejected_lp:
        correct += 1
print(f"reward accuracy: {correct}/{len(preference_data)} = {correct/len(preference_data):.0%}")
print("(in real eval: use held-out set + real LLM)")
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`, `import torch.nn.functional as F`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Try `beta=0.05` vs `beta=0.5` — does the policy become more or less conservative?
> 2. Replace DPO with IPO (identity preference optimisation) — does it reduce overfitting?
> 3. Use an LLM-as-judge to label preferences on a dataset, then DPO on the labels.
> 4. Run DPO on top of an SFT model and compare quality to SFT-only.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement DPO from scratch with `transformers.Trainer` (no `DPOTrainer`).
> 2. Build an LLM-as-judge labelling pipeline to generate preference data.
> 3. Implement KTO (Kahneman-Tversky Optimisation) — uses unpaired preference signals.
> 4. Compare DPO and PPO (RLHF) on a small benchmark.
> 5. Implement a length-bias check: penalise the model if it learns to prefer short/long responses.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What problem does DPO solve, and how does it differ from RLHF/PPO?
> 2. What is the role of the reference model in DPO?
> 3. What does `beta` control, and how do you choose it?
> 4. What is the DPO loss, in plain English?
> 5. What is reward accuracy, and how do you measure it?
> 6. What are the common failure modes of DPO?
> 7. How does DPO relate to IPO and KTO?
> 8. When would you choose DPO over RLHF?
