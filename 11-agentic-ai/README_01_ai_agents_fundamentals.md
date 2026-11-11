# Code Walkthrough — `01_ai_agents_fundamentals.ipynb`

> Module: **11-agentic-ai** | Notebook: **01_ai_agents_fundamentals.ipynb**
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

> # 01 — AI Agents Fundamentals
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
> ## 1. Concept Overview — AI agents — from prompts to actions

> An AI agent is an LLM in a loop: it perceives state, decides on an action
> (potentially calling tools), observes the result, and repeats until it
> reaches a goal. The LLM is the *reasoner*; the surrounding loop is the
> *agent*.
> 
> **Beginner track:** understand the perceive-reason-act loop. Every agent
> framework is a variation on it.
> **Advanced track:** also learn finite-state-machine agents, plan-and-execute
> patterns, and the trade-offs between ReAct and chain-of-thought planning.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> An agent is a function $\pi: \text{state} \to \text{action}$ where $\pi$ is
> realised by an LLM. The state evolves via $s_{t+1} = T(s_t, a_t)$ where $T$
> is the environment. The agent halts when an LLM-produced stop signal fires
> (e.g. `Final Answer:`).
> 
> Total tokens used: $\sum_t |\text{prompt}_t| + |\text{action}_t| + |\text{obs}_t|$ —
> the agent's *cost*. Long-running agents are dominated by context cost.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements the ReAct (Reason + Act) loop from scratch
> in pure Python — no LangChain, no LLM. We use a *mock* LLM that emits
> canned actions to show the loop structure.
> 
> **Why it matters:** every agent framework (LangChain, LangGraph, AutoGen,
> CrewAI) implements this same loop. Seeing it in 50 lines of pure Python
> demystifies them all.
> 
> **What to watch for:** the `step()` method always returns one of three
> things: an action to execute, a final answer, or a request to stop. The
> loop runs until one of those fires.


### Code cell #1
```python
import re

# A mock LLM that, given a prompt, returns a "thought" and an "action".
class MockLLM:
    """Returns canned ReAct-formatted responses for demo purposes."""
    def __call__(self, prompt):
        if "search" in prompt.lower() and "Step 1" not in prompt:
            return ('Thought: I need to search for the population.\n'
                    'Action: search\n'
                    'Action Input: "population of France"\n')
        if "67 million" in prompt:
            return ('Thought: Now I need to multiply by 3.\n'
                    'Action: calculator\n'
                    'Action Input: 67 * 3\n')
        if "201" in prompt:
            return ('Thought: I have the answer.\n'
                    'Final Answer: The population of France times 3 is about 201 million.\n')
        return "Thought: I don't know what to do.\nFinal Answer: I cannot help."

# Mock tools: their "execution" is just a canned string.
def tool_search(query):
    return "France has a population of about 67 million people."

def tool_calculator(expression):
    try:
        return str(eval(expression))   # safe in this controlled demo
    except Exception as e:
        return f"error: {e}"

TOOLS = {"search": tool_search, "calculator": tool_calculator}

# The ReAct agent loop.
def react_agent(question, llm, max_steps=5):
    scratchpad = ""
    for step in range(max_steps):
        prompt = f"Question: {question}\n{scratchpad}\nStep {step+1}:"
        out = llm(prompt)
        # Parse the LLM output for an action or final answer.
        if "Final Answer:" in out:
            return out.split("Final Answer:")[1].strip(), step+1
        m = re.search(r"Action: (\w+)\nAction Input: (.+)", out)
        if not m:
            return f"(no action parsed) {out}", step+1
        tool_name, tool_input = m.groups()
        tool_input = tool_input.strip().strip('"')
        if tool_name not in TOOLS:
            return f"(unknown tool {tool_name})", step+1
        observation = TOOLS[tool_name](tool_input)
        scratchpad += f"\n{out}\nObservation: {observation}"
    return "(max steps reached)", max_steps

answer, steps = react_agent("What is the population of France times 3?", MockLLM())
print(f"answer after {steps} step(s): {answer}")
```

**What this code does:**

- **Headline:** `import re`
- **Defines:** class `MockLLM` — Returns canned ReAct-formatted responses for demo purposes.; def `tool_search`; def `tool_calculator`; def `react_agent`
- **Key imports:** `import re`

> **Takeaway:** An agent is just a `while` loop around an LLM. The framework
> code you don't see is doing exactly this — parsing actions, calling tools,
> appending observations.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-implements the same agent using `langchain`'s
> `create_react_agent` executor and `AgentExecutor`. If `langchain` is not
> installed, the demo is skipped gracefully.
> 
> **Why it matters:** LangChain is the dominant Python agent framework.
> Knowing what it does *under the hood* (our scratch implementation above)
> lets you debug it when it inevitably produces weird output.
> 
> **What to watch for:** the `agent_executor.invoke({"input": ...})` API is
> the post-0.2 LangChain convention. The older `run(...)` API is deprecated.


### Code cell #2
```python
try:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    # In production you'd use a real LLM. We skip the live demo to avoid
    # requiring an API key.
    print("langchain is installed; pattern would be:")
    print("  llm = ChatOpenAI(model='gpt-4o-mini')")
    print("  tools = [Tool(name='search', func=..., description='...')]")
    print("  agent = create_react_agent(llm, tools, prompt)")
    print("  executor = AgentExecutor(agent=agent, tools=tools)")
    print("  result = executor.invoke({'input': question})")
except ImportError as e:
    print(f"langchain not installed ({e}) — pip install langchain langchain-openai")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from langchain.agents import create_react_agent`, `from langchain.tools import Tool`, `from langchain_openai import ChatOpenAI`, `from langchain.prompts import ChatPromptTemplate`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Defines a tiny agent benchmark — 5 multi-step questions
> that need a tool call to answer. We'll use this in section 7 to evaluate
> our agent.
> 
> **Why it matters:** agent evaluation needs tasks where the answer is not
> in the LLM's parametric memory. Pure-knowledge questions ("What is X?")
> are useless for agent eval — they don't exercise the loop.
> 
> **What to watch for:** each question has a *gold answer* and a *required
> tool*. The agent eval should check both: did you call the right tool, and
> did you get the right answer?


### Code cell #3
```python
agent_eval = [
    {"q": "What is 12 * 7?",            "tool": "calculator", "gold": "84"},
    {"q": "What is the population of France times 3?",
     "tool": "search+calculator", "gold": "201 million"},
    {"q": "Square root of 144?",        "tool": "calculator", "gold": "12"},
    {"q": "What is 50 * (3 + 2)?",       "tool": "calculator", "gold": "250"},
    {"q": "How tall is the Eiffel Tower in metres divided by 100?",
     "tool": "search+calculator", "gold": "3.3"},
]
print(f"agent eval size: {len(agent_eval)}")
for qa in agent_eval:
    print(f"  Q: {qa['q']}")
    print(f"    tool: {qa['tool']}  gold: {qa['gold']}")
```

**What this code does:**

- **Headline:** `agent_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots the agent's token cost per step on a synthetic
> trace. The x-axis is step number; the y-axis is the cumulative tokens in
> the prompt (because the scratchpad grows).
> 
> **Why it matters:** the *growing prompt* is the dominant cost of any agent.
> Step 5's prompt is much larger than step 1's, because it contains the full
> history. This is why context-window management is critical for long agents.
> 
> **What to watch for:** the curve is roughly linear in steps for short
> agents, but quadratic if you re-send the entire history each time. Some
> frameworks use sliding windows or summarisation to cap this.


### Code cell #4
```python
import matplotlib.pyplot as plt
import numpy as np

# Mock trace: 5 steps, each appending ~50 tokens to the scratchpad.
steps = list(range(1, 6))
token_per_step = 80  # base prompt
cumulative = []
for step in steps:
    token_per_step += 50  # scratchpad grows by ~50 tokens per step
    cumulative.append(token_per_step)

plt.figure(figsize=(7, 4))
plt.plot(steps, cumulative, "o-", color="steelblue")
plt.xlabel("step"); plt.ylabel("tokens in prompt")
plt.title("Agent prompt-size growth (ReAct loop)")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print(f"total tokens used: {sum(cumulative)}")
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import numpy as np`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Agents are evaluated by:
> 
> - **Task success rate** — did the agent produce the right answer?
> - **Tool-call accuracy** — did it call the right tools?
> - **Step efficiency** — how many steps did it take (fewer is better)?
> - **Cost** — total tokens (and therefore $) per task.

> **What this does:** A tiny agent evaluator: runs the agent on each eval
> question and reports success rate, average steps, and tool-call accuracy.
> 
> **Why it matters:** without an agent evaluator, you can't tell whether your
> prompt changes *help* or *hurt*. This is the minimum tool for any agent
> engineering work.
> 
> **What to watch for:** we score with substring match — fine for these
> demo answers, but real eval needs semantic match (LLM-as-judge).


### Code cell #5
```python
# Run our mock agent on the eval set and report metrics.
successes = 0
total_steps = 0
for qa in agent_eval:
    answer, steps = react_agent(qa["q"], MockLLM())
    # Crude substring match — real eval would use semantic match.
    ok = qa["gold"].lower() in answer.lower()
    successes += int(ok)
    total_steps += steps
    print(f"  Q: {qa['q'][:40]:42s}  ok={ok}  steps={steps}")

n = len(agent_eval)
print(f"\nsuccess rate: {successes}/{n} = {successes/n:.0%}")
print(f"avg steps   : {total_steps/n:.1f}")
```

**What this code does:**

- **Headline:** `successes = 0`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Allow the agent up to 10 steps instead of 5 — does it solve harder tasks?
> 2. Add a `summarise` tool that compresses the scratchpad — does it reduce token cost?
> 3. Switch the mock LLM to a real one (OpenAI, Anthropic, or open-source via `transformers`).
> 4. Try the same task with `create_react_agent` (LangChain) and compare output.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a *plan-and-execute* agent: first generate a plan, then execute each step.
> 2. Implement a *reflection* step: after the final answer, ask the LLM to critique and revise.
> 3. Build a tool that lets the agent call *itself* (recursive sub-agent) for sub-tasks.
> 4. Implement a context-window manager that summarises the scratchpad when it exceeds N tokens.
> 5. Build an agent evaluator that uses LLM-as-judge for semantic answer matching.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is an AI agent, and how does it differ from a plain LLM call?
> 2. What are the components of the ReAct pattern?
> 3. Why does the scratchpad grow with each step, and why does that matter?
> 4. How would you prevent an agent from looping indefinitely?
> 5. What is the difference between ReAct and chain-of-thought?
> 6. How would you evaluate an agent?
> 7. What is the cost of an agent in tokens, and how do you reduce it?
> 8. When would you use a finite-state-machine agent instead of ReAct?
