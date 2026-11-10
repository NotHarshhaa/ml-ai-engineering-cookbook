# Code Walkthrough — `01_prompt_engineering.ipynb`

> Module: **10-generative-ai** | Notebook: **01_prompt_engineering.ipynb**
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

> # 01 — Prompt Engineering Patterns
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
> ## 1. Concept Overview — Prompt engineering patterns

> Prompt engineering is the practice of *programming* an LLM with natural
> language. Instead of editing weights, you craft an instruction + context
> that guides the model toward the behaviour you want.
> 
> **Beginner track:** learn the four core patterns — zero-shot, few-shot,
> chain-of-thought, role-play. That covers most production prompts.
> **Advanced track:** also learn ReAct (reasoning + acting),
> tree-of-thoughts, and self-consistency for harder tasks.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> An LLM defines a conditional distribution $P(y \mid x)$ over completions $y$
> given a prompt $x$. Prompt engineering is *control* over $P$ via the choice
> of $x$. The transformer doesn't change; only the input does.
> 
> Few-shot: $x = (\text{instruction}, (x_1, y_1), \ldots, (x_k, y_k), x_\text{new})$.
> Chain-of-thought: $y = (\text{reasoning}, \text{answer})$, with the model
> explicitly producing intermediate steps.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** A `prompt` helper that formats different prompt
> patterns into a single string. The function itself doesn't call an LLM —
> we use it to *see* what each pattern produces.
> 
> **Why it matters:** once you can build prompts as strings, you can debug
> them. Most prompt failures are actually template failures.
> 
> **What to watch for:** the `few_shot` pattern puts examples *before* the
> new question — never after. Reversing it makes the LLM treat the new
> question as an example to copy.


### Code cell #1
```python
def zero_shot(instruction, question):
    return f"{instruction}\n\nQ: {question}\nA:"

def few_shot(instruction, examples, question):
    # examples is a list of (q, a) tuples.
    ex_text = "".join(f"\nQ: {q}\nA: {a}" for q, a in examples)
    return f"{instruction}{ex_text}\n\nQ: {question}\nA:"

def chain_of_thought(instruction, question):
    # Force the model to reason step by step before the final answer.
    cot_instruction = instruction + " Let's think step by step."
    return f"{cot_instruction}\n\nQ: {question}\nA:"

instr = "You are a careful math tutor. Answer the question precisely."
q = "If a train travels 60 mph for 2.5 hours, how far does it go?"

print("=== ZERO-SHOT ===")
print(zero_shot(instr, q))
print("\n=== FEW-SHOT ===")
examples = [("What is 5 + 7?", "12"), ("What is 9 * 8?", "72")]
print(few_shot(instr, examples, q))
print("\n=== CHAIN-OF-THOUGHT ===")
print(chain_of_thought(instr, q))
```

**What this code does:**

- **Headline:** `def zero_shot(instruction, question):`
- **Defines:** def `zero_shot`; def `few_shot`; def `chain_of_thought`

> **Takeaway:** A prompt is just a string. The "engineering" part is *picking
> the right template* for the task — zero-shot for simple, few-shot for
> formatting, chain-of-thought for reasoning.

> **What this does:** Implements the ReAct (Reasoning + Acting) prompt
> template — the model alternates between `Thought:`, `Action:`, and
> `Observation:` lines to interleave reasoning with tool calls.
> 
> **Why it matters:** ReAct is the foundation of modern *agentic AI* (the
> next module). It's the standard pattern for LLMs that call external tools.
> 
> **What to watch for:** the `Observation:` lines come from the tool, not
> the model. The prompt is designed to *stop* after `Action:` and let the
> framework call the tool.


### Code cell #2
```python
def react_prompt(question, tools):
    """Build a ReAct prompt: question, available tools, scratchpad."""
    tools_desc = "\n".join(
        f"- {name}: {desc}" for name, desc in tools.items()
    )
    return f"""Answer the following question using the tools below.

Available tools:
{tools_desc}

Use this format:
Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>
(then a tool will produce: Observation: <tool output>)
Thought: <your next reasoning based on the observation>
Final Answer: <the answer>

Question: {question}
Thought: """

tools = {
    "calculator": "Performs arithmetic. Input: an arithmetic expression as text.",
    "search": "Searches the web. Input: a search query.",
}
print(react_prompt("What is the population of France times 3?", tools))
```

**What this code does:**

- **Headline:** `def react_prompt(question, tools):`
- **Defines:** def `react_prompt` — Build a ReAct prompt: question, available tools, scratchpad.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Calls a small HuggingFace LLM (`gpt2`) with three
> different prompt templates and shows the effect on output. Falls back to
> a `mock` if transformers isn't installed.
> 
> **Why it matters:** the prompt is the single biggest lever on LLM quality.
> Seeing the same model produce wildly different outputs for different
> prompts is the core insight of prompt engineering.
> 
> **What to watch for:** GPT-2 is *tiny* (~125M params) and won't follow
> instructions the way GPT-4 does. The point here is to *see* the prompt
> structure, not to get good outputs.


### Code cell #3
```python
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()

    def generate(prompt, max_new=40):
        ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                 top_p=0.9, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    prompts = [
        ("zero-shot",  zero_shot("You are a math tutor.", "What is 12 * 7?")),
        ("few-shot",   few_shot("You are a math tutor.",
                                [("What is 2+3?", "5"), ("What is 4*5?", "20")],
                                "What is 12 * 7?")),
        ("CoT",        chain_of_thought("You are a math tutor.", "What is 12 * 7?")),
    ]
    for name, p in prompts:
        print(f"\n=== {name} ===")
        print(generate(p))
        print("---")
except Exception as e:
    print(f"skipped LLM demo ({e})")
    print("(install `transformers` to see the live demo)")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from transformers import GPT2Tokenizer`, `import torch`
- **Notable calls:** `GPT2Tokenizer`, `GPT2LMHeadModel`, `model.generate`, `model.eval`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small evaluation dataset for prompt
> experiments: 10 short math questions with known answers. We'll use it
> to compare prompt patterns.
> 
> **Why it matters:** prompt engineering is empirical. To know which prompt
> pattern works best, you need a small evaluation set — not a vibe check.
> 
> **What to watch for:** the questions are deliberately small (one-line
> answers). Real prompt benchmarks (GSM8K, MMLU) are much harder and need
> thousands of examples for statistical power.


### Code cell #4
```python
eval_set = [
    ("What is 12 * 7?", "84"),
    ("What is 144 / 12?", "12"),
    ("What is 9 + 10?", "19"),
    ("What is 100 - 37?", "63"),
    ("What is 15 * 4?", "60"),
    ("What is 81 / 9?", "9"),
    ("What is 25 + 25?", "50"),
    ("What is 11 * 11?", "121"),
    ("What is 200 / 4?", "50"),
    ("What is 8 * 9?", "72"),
]
print(f"eval set size: {len(eval_set)}")
print("first 3:", eval_set[:3])
```

**What this code does:**

- **Headline:** `eval_set = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots a mock accuracy comparison of three prompt
> patterns on our eval set. Since `distilgpt2` can't follow instructions
> well, we use *mock* numbers that match what real LLMs (GPT-3.5+)
> actually achieve on GSM8K-style tasks.
> 
> **Why it matters:** the *shape* of this comparison is the universal
> prompt-engineering result: zero-shot is worst, few-shot is better, CoT
> is best for reasoning tasks.
> 
> **What to watch for:** CoT helps most for *multi-step* reasoning. For
> single-step tasks (e.g. sentiment classification) the gap is smaller.


### Code cell #5
```python
import matplotlib.pyplot as plt

# Mock accuracies on a small math eval set — these match published trends on GSM8K.
patterns = ["zero-shot", "few-shot (3 ex)", "chain-of-thought"]
mock_accuracies = [0.30, 0.55, 0.80]

plt.figure(figsize=(7, 4))
bars = plt.bar(patterns, mock_accuracies, color=["#e74c3c", "#3498db", "#27ae60"],
               edgecolor="white")
plt.ylim(0, 1.0)
plt.ylabel("accuracy on 10-question math eval")
plt.title("Mock comparison: prompt patterns on small math eval")
for bar, acc in zip(bars, mock_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{acc:.0%}", ha="center", fontsize=10)
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Prompts are evaluated by:
> 
> - **Task accuracy** on a benchmark (GSM8K, MMLU, HumanEval).
> - **Format compliance** — did the output match the requested format?
> - **Robustness** — does the prompt work across multiple model sizes?
> - **Cost** — longer prompts cost more tokens; does the gain justify the cost?

> **What this does:** A simple `evaluate_prompt` function that runs a prompt
> template against an eval set and computes the accuracy. We provide a mock
> scorer for the case where no LLM is available.
> 
> **Why it matters:** this is the *minimum* prompt-engineering experiment —
> a template + an eval set + a scorer. Without it, you're doing vibes, not
> engineering.
> 
> **What to watch for:** the scorer strips whitespace and lowercases — exact
> string match. Real evaluation often needs semantic match (LLM-as-judge).


### Code cell #6
```python
def evaluate_prompt(template_fn, eval_set, generate_fn=None):
    """Apply template to each question, generate, and score exact match."""
    correct = 0
    for q, gold in eval_set:
        prompt = template_fn(q)
        if generate_fn is None:
            # Mock generator: always return a plausible answer.
            pred = gold
        else:
            pred = generate_fn(prompt)
        # Simple string-match scorer.
        if str(pred).strip().lower().startswith(str(gold).strip().lower()):
            correct += 1
    return correct / len(eval_set)

# Mock evaluation: pretend each prompt template succeeds.
mock_acc = evaluate_prompt(lambda q: zero_shot("You are a math tutor.", q),
                            eval_set, generate_fn=None)
print(f"mock accuracy (zero-shot, mock generator): {mock_acc:.0%}")
```

**What this code does:**

- **Headline:** `def evaluate_prompt(template_fn, eval_set, generate_fn=None):`
- **Defines:** def `evaluate_prompt` — Apply template to each question, generate, and score exact match.

> ---
> 
> ## 8. Experiments — Experiments

> 1. Try 1-shot vs 3-shot vs 5-shot — does more examples always help?
> 2. Try `temperature=0` (greedy) vs `temperature=1.0` (random) — which is more reliable?
> 3. Add `Let's verify step by step.` after the chain-of-thought answer.
> 4. Try a *negative* instruction (`Don't do X`) and check whether LLMs respect it.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Build a `PromptTemplate` class with `.format(**kwargs)` and validation.
> 2. Implement *self-consistency*: sample N times, take the majority answer.
> 3. Implement tree-of-thoughts on a small reasoning task.
> 4. Build a prompt-evaluation pipeline that runs on a small open-source LLM.
> 5. Implement a *prompt compression* technique (e.g. LLMLingua-style).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between zero-shot, few-shot, and chain-of-thought prompting?
> 2. Why does chain-of-thought help with reasoning tasks?
> 3. What is the ReAct pattern, and when would you use it?
> 4. Why are prompts sensitive to formatting (newlines, capitalisation)?
> 5. How would you evaluate a prompt template systematically?
> 6. What is prompt injection, and how do you mitigate it?
> 7. What is the difference between instruction tuning and in-context learning?
> 8. Why does few-shot learning sometimes hurt performance?
