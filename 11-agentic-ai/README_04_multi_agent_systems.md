# Code Walkthrough — `04_multi_agent_systems.ipynb`

> Module: **11-agentic-ai** | Notebook: **04_multi_agent_systems.ipynb**
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

> # 04 — Multi-Agent Systems
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
> ## 1. Concept Overview — Multi-agent systems

> A multi-agent system (MAS) is several agents that collaborate on a task.
> Each agent has a role (researcher, writer, critic, executor), and the
> framework routes messages between them. Frameworks: AutoGen, CrewAI,
> LangGraph's multi-agent patterns.
> 
> **Beginner track:** learn the supervisor pattern: one agent routes tasks
> to specialist agents and combines results.
> **Advanced track:** also learn peer-to-peer agent communication,
> debate patterns, and the challenges of consistency across agents.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A multi-agent system is a set of agents $\{A_1, \ldots, A_n\}$ plus a
> communication protocol. Each agent is itself an LLM-driven function. The
> system's behaviour emerges from the message-passing graph.
> 
> Common patterns:
> - **Supervisor**: $A_\text{sup}$ routes sub-tasks to $A_1, \ldots, A_n$ and
>   aggregates.
> - **Hierarchical**: a tree of supervisors.
> - **Peer-to-peer / debate**: agents exchange messages until consensus.
> - **Pipeline / sequential**: $A_1 \to A_2 \to \ldots \to A_n$.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a 3-agent supervisor pattern in pure Python:
> a Researcher (uses search), a Writer (drafts), a Critic (reviews). The
> Supervisor routes between them based on the current state.
> 
> **Why it matters:** multi-agent systems look magical until you've built one.
> Then they look like a state machine where each node is an LLM call. This is
> the seed of every production multi-agent framework.
> 
> **What to watch for:** each agent takes the *full conversation history* —
> they're stateless functions of (task, history) → message. The framework
> manages the history.


### Code cell #1
```python
from typing import Callable, List

class Agent:
    """A single agent: a named LLM-callable that takes a task + history."""
    def __init__(self, name: str, system_prompt: str, llm: Callable):
        self.name = name
        self.system = system_prompt
        self.llm = llm
    def __call__(self, task, history):
        prompt = f"{self.system}\n\nTask: {task}\n\nHistory: {history}"
        return self.llm(prompt)

# Mock LLM that returns canned responses per agent.
def mock_researcher_llm(prompt):
    return "Researcher: I found 3 sources on this topic."

def mock_writer_llm(prompt):
    if "Researcher" in prompt:
        return "Writer: Here is a draft based on the research."
    return "Writer: Here is a draft."

def mock_critic_llm(prompt):
    if "draft" in prompt.lower() and "good" not in prompt.lower():
        return "Critic: The draft is good. APPROVE."
    return "Critic: Please revise."

researcher = Agent("Researcher", "You research topics thoroughly.", mock_researcher_llm)
writer     = Agent("Writer",     "You write clear drafts.",       mock_writer_llm)
critic     = Agent("Critic",     "You critique drafts. Say APPROVE when done.", mock_critic_llm)

def supervisor(task, max_rounds=5):
    history = []
    # Pipeline: research -> write -> critique, loop on critique.
    for round in range(max_rounds):
        # Step 1: research.
        msg = researcher(task, "\n".join(history))
        history.append(msg)
        # Step 2: write.
        msg = writer(task, "\n".join(history))
        history.append(msg)
        # Step 3: critique.
        msg = critic(task, "\n".join(history))
        history.append(msg)
        if "APPROVE" in msg:
            return history, round + 1
    return history, max_rounds

history, rounds = supervisor("Write an article about transformers.")
print(f"completed in {rounds} round(s):")
for m in history:
    print(f"  {m}")
```

**What this code does:**

- **Headline:** `from typing import Callable, List`
- **Defines:** class `Agent` — A single agent: a named LLM-callable that takes a task + history.; def `mock_researcher_llm`; def `mock_writer_llm`; def `mock_critic_llm`; def `supervisor`
- **Key imports:** `from typing import Callable`

> **Takeaway:** Multi-agent = agents + a routing function. The supervisor is
> just a loop that decides which agent to call next. Real frameworks add
> persistence, retries, and parallelism.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Shows the equivalent in `crewai` — the popular Python
> multi-agent framework. If it's not installed, we print the pattern.
> 
> **Why it matters:** CrewAI's API matches the supervisor pattern exactly:
> you define agents with roles, give them tasks, and a Crew runs them in
> sequence or parallel. Knowing the underlying loop makes the API obvious.
> 
> **What to watch for:** CrewAI tasks have `expected_output` and `agent` —
> the order of tasks in the list defines the execution order (sequential
> by default).


### Code cell #2
```python
try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew
    # We don't construct a real crew here because it requires a live LLM.
    print("crewai is installed; the pattern would be:")
    print("  researcher = CrewAgent(role='Researcher', goal='...', backstory='...', llm=...)")
    print("  writer     = CrewAgent(role='Writer', goal='...', backstory='...', llm=...)")
    print("  critic     = CrewAgent(role='Critic', goal='...', backstory='...', llm=...)")
    print("  tasks = [CrewTask(description=..., expected_output=..., agent=researcher), ...]")
    print("  crew = Crew(agents=[researcher, writer, critic], tasks=tasks)")
    print("  result = crew.kickoff()")
except ImportError as e:
    print(f"crewai not installed ({e}) — pip install crewai")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from crewai import Agent as CrewAgent`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Defines 3 multi-agent tasks of varying difficulty to
> evaluate the supervisor pattern.
> 
> **Why it matters:** multi-agent systems are *expensive* (each round is N
> LLM calls). The benchmark should include tasks where the multi-agent
> structure actually helps vs a single agent.
> 
> **What to watch for:** the easy task doesn't need 3 agents; the multi-agent
> system is overkill. The hard task (long-form writing) is where multi-agent
> shines.


### Code cell #3
```python
mas_eval = [
    {"task": "Write a one-sentence summary of Python.", "needs_multi": False},
    {"task": "Write a 3-paragraph article about transformers.", "needs_multi": True},
    {"task": "Write a 1-page essay on AI safety, with citations.", "needs_multi": True},
]
print(f"MAS eval size: {len(mas_eval)}")
for t in mas_eval:
    print(f"  Task: {t['task']}")
    print(f"    multi-agent needed? {t['needs_multi']}")
```

**What this code does:**

- **Headline:** `mas_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the supervisor pattern: a central supervisor
> routes messages between specialist agents, who report back.
> 
> **Why it matters:** the supervisor diagram is the canonical way to present
> multi-agent systems. It shows the flow of control clearly.
> 
> **What to watch for:** the supervisor is itself an agent (LLM call). In a
> real system the supervisor decides routing dynamically based on the
> current state, not a fixed sequence.


### Code cell #4
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 8); ax.set_ylim(0, 6); ax.axis("off")

# Nodes: supervisor in centre, 3 specialists around.
nodes = [
    ("Supervisor", 4, 5, "#3498db"),
    ("Researcher", 1.5, 2, "#27ae60"),
    ("Writer",     4,   2, "#9b59b6"),
    ("Critic",     6.5, 2, "#e74c3c"),
    ("Output",     4,   0.3, "#7f8c8d"),
]
for label, x, y, color in nodes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.7, y-0.25), 1.4, 0.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="white", alpha=0.85))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

# Edges: supervisor -> specialists, specialists -> supervisor, supervisor -> output.
edges = [(4, 4.75, 1.5, 2.25), (4, 4.75, 4, 2.25), (4, 4.75, 6.5, 2.25),
         (1.5, 2.25, 4, 4.75), (4, 2.25, 4, 4.75), (6.5, 2.25, 4, 4.75),
         (4, 4.75, 4, 0.55)]
for x1, y1, x2, y2 in edges:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6))

ax.set_title("Multi-agent supervisor pattern")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import matplotlib.patches as mpatches`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Multi-agent systems are evaluated by:
> 
> - **Final output quality** — judged by an LLM-as-judge or human rater.
> - **Round count** — how many rounds to converge? Fewer is cheaper.
> - **Convergence rate** — does it stop, or loop forever?
> - **Cost per task** — total tokens (sum over all agents and rounds).

> **What this does:** Runs the supervisor on the eval set and reports
> rounds-to-convergence and final-output length.
> 
> **Why it matters:** rounds-to-convergence is the cost metric for iterative
> multi-agent systems. If your critic never approves, you have a design
> problem.
> 
> **What to watch for:** we use mock agents so convergence is deterministic.
> Real LLMs may loop, which is why production MAS frameworks have max-round
> limits and timeout policies.


### Code cell #5
```python
for qa in mas_eval:
    history, rounds = supervisor(qa["task"])
    final_msg = history[-1] if history else ""
    print(f"  Task: {qa['task'][:50]:52s}")
    print(f"    rounds: {rounds}  final length: {len(final_msg)} chars")
```

**What this code does:**

- **Headline:** `for qa in mas_eval:`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a "fact-checker" agent that verifies the writer's claims.
> 2. Make the supervisor LLM-driven: it decides which agent to call next based on state.
> 3. Run the writer and researcher in parallel — does it speed things up?
> 4. Replace mock LLMs with real ones and run on the eval set.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a *debate* pattern: 2 agents argue, a 3rd judges.
> 2. Build a hierarchical MAS: a supervisor-of-supervisors.
> 3. Implement a consensus protocol: agents must agree before producing output.
> 4. Add a "human-in-the-loop" agent that pauses for user input.
> 5. Implement a multi-agent RAG: each agent retrieves from a different source.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is a multi-agent system, and when is it preferable to a single agent?
> 2. What is the supervisor pattern, and what are its alternatives?
> 3. How do you prevent infinite loops in a multi-agent system?
> 4. What is the cost of a multi-agent system compared to a single agent?
> 5. How would you evaluate a multi-agent system?
> 6. What is agent debate, and when does it improve output quality?
> 7. How would you handle conflicting outputs from different agents?
> 8. What is the difference between CrewAI, AutoGen, and LangGraph's multi-agent?
