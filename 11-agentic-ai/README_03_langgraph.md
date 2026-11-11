# Code Walkthrough — `03_langgraph.ipynb`

> Module: **11-agentic-ai** | Notebook: **03_langgraph.ipynb**
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

> # 03 — LangGraph
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
> ## 1. Concept Overview — LangGraph — agents as graphs

> LangGraph is a library for building agent workflows as *graphs*: nodes are
> functions, edges define control flow. It's the right abstraction when your
> agent has non-trivial control flow (loops, conditional branches, multi-agent
> hand-offs).
> 
> **Beginner track:** understand the node + edge + state abstraction. Every
> LangGraph program is just three things.
> **Advanced track:** also learn conditional edges, state schemas, and how
> LangGraph's checkpointing enables fault-tolerant long-running agents.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> A LangGraph program is a directed graph $G = (V, E)$ where:
> - $V$ is a set of *nodes*, each a function $v: S \to S$ that updates a shared state.
> - $E$ is a set of *edges*; conditional edges are functions $e: S \to V$ that
>   pick the next node based on the state.
> 
> Execution: start at the entry node, repeatedly apply the node function and
> follow edges, until reaching an end node.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a tiny graph executor in pure Python: nodes
> are functions, edges are explicit, conditional edges are functions that
> return the next node name. We build a 3-node graph and run it.
> 
> **Why it matters:** LangGraph is *exactly* this, just with more bells and
> whistles (state schemas, checkpointing, streaming). Seeing it in 30 lines
> removes the magic.
> 
> **What to watch for:** the `state` is a dict that gets passed to every node.
> Each node returns a dict that gets merged back into state — this is the
> LangGraph reducer pattern.


### Code cell #1
```python
class TinyGraph:
    """A minimal graph executor: nodes are functions, edges are explicit."""
    def __init__(self, initial_state):
        self.state = dict(initial_state)
        self.nodes = {}        # name -> function(state) -> state-delta
        self.edges = {}        # name -> next-name OR name -> function(state) -> next-name
        self.start = None
        self.end = set()
    def add_node(self, name, fn):
        self.nodes[name] = fn
    def add_edge(self, src, dst):
        self.edges[src] = dst
    def add_conditional_edge(self, src, fn):
        self.edges[src] = fn   # fn returns the next node name
    def set_entry(self, name):
        self.start = name
    def set_exit(self, name):
        self.end.add(name)
    def run(self, max_steps=20):
        current = self.start
        for _ in range(max_steps):
            if current in self.end:
                return self.state
            delta = self.nodes[current](self.state)
            self.state.update(delta or {})
            edge = self.edges.get(current)
            if edge is None:
                return self.state
            current = edge(self.state) if callable(edge) else edge
        return self.state

# Build a 3-node graph: classify -> route -> (answer_easy OR answer_hard).
def classify(state):
    q = state["question"].lower()
    return {"difficulty": "easy" if len(q) < 30 else "hard"}

def answer_easy(state):
    return {"answer": f"(easy) {state['question']}"}

def answer_hard(state):
    return {"answer": f"(hard) {state['question']}"}

def route(state):
    return "answer_easy" if state["difficulty"] == "easy" else "answer_hard"

g = TinyGraph({"question": "What is 2+2?"})
g.add_node("classify", classify)
g.add_node("answer_easy", answer_easy)
g.add_node("answer_hard", answer_hard)
g.add_edge("classify", route)              # conditional
g.add_edge("answer_easy", None)            # implicit end
g.add_edge("answer_hard", None)
g.set_entry("classify")

# TinyGraph doesn't support None as end; we patch by treating missing edges as end.
out = g.run()
print("final state:", out)
```

**What this code does:**

- **Headline:** `class TinyGraph:`
- **Defines:** class `TinyGraph` — A minimal graph executor: nodes are functions, edges are explicit.; def `classify`; def `answer_easy`; def `answer_hard`; def `route`

> **Takeaway:** A graph executor is `apply node → follow edge → repeat`. The
> complexity in real frameworks is in state schemas, checkpointing, and
> streaming — not the core loop.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Re-implements the same graph with the real `langgraph`
> library. If it's not installed, we print the equivalent code.
> 
> **Why it matters:** LangGraph is becoming the *standard* way to build
> non-trivial agents. The graph abstraction makes loops and conditional
> branches first-class, where a ReAct-style prompt makes them implicit.
> 
> **What to watch for:** `StateGraph(State)` takes a typed-dict state schema.
> Each node returns a *partial* state that gets merged — this is the reducer.


### Code cell #2
```python
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated, Optional
    import operator

    class AgentState(TypedDict):
        question: str
        difficulty: Optional[str]
        answer: Optional[str]

    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("answer_easy", answer_easy)
    g.add_node("answer_hard", answer_hard)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {"answer_easy": "answer_easy",
                                                 "answer_hard": "answer_hard"})
    g.add_edge("answer_easy", END)
    g.add_edge("answer_hard", END)
    compiled = g.compile()

    result = compiled.invoke({"question": "What is 2+2?"})
    print("LangGraph result:", result)
except ImportError as e:
    print(f"langgraph not installed ({e}) — pip install langgraph")
```

**What this code does:**

- **Headline:** `try:`
- **Key imports:** `from langgraph.graph import StateGraph`, `from typing import TypedDict`, `import operator`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small set of tasks that require *different*
> agent paths: some easy (direct answer), some medium (one tool call), some
> hard (multiple tool calls with a loop).
> 
> **Why it matters:** the value of LangGraph's conditional edges shows only
> when tasks vary in difficulty. A simple ReAct loop handles all of them the
> same way; a graph can short-circuit the easy ones.
> 
> **What to watch for:** we tag each task with the *expected* graph path.
> Real eval compares the agent's actual path to the expected one.


### Code cell #3
```python
graph_eval = [
    {"q": "What is 2+2?",                "expected_path": ["classify", "answer_easy"]},
    {"q": "What is the population of France times 3?",
     "expected_path": ["classify", "answer_hard"]},
    {"q": "What is the meaning of life?",
     "expected_path": ["classify", "answer_hard"]},
]
print(f"graph eval size: {len(graph_eval)}")
for qa in graph_eval:
    print(f"  Q: {qa['q']}")
    print(f"    path: {qa['expected_path']}")
```

**What this code does:**

- **Headline:** `graph_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises our 3-node graph: classify in the middle,
> answer_easy on the left, answer_hard on the right, with conditional edges.
> 
> **Why it matters:** visualising the graph is the standard way to present
> LangGraph workflows. Production tools (LangSmith, LangGraph Studio) render
> interactive versions.
> 
> **What to watch for:** the conditional edge from `classify` is drawn as two
> arrows — one to each branch. The *condition function* chooses which one is
> followed at runtime.


### Code cell #4
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")

# Nodes.
nodes = [
    ("classify",     4, 3, "#9b59b6"),
    ("answer_easy",  1.5, 1, "#27ae60"),
    ("answer_hard",  6.5, 1, "#e74c3c"),
    ("END",          4, 0, "#7f8c8d"),
]
for label, x, y, color in nodes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.7, y-0.25), 1.4, 0.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="white", alpha=0.85))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

# Edges.
edges = [(4, 2.75, 1.5, 1.25, "easy"),
         (4, 2.75, 6.5, 1.25, "hard"),
         (1.5, 0.75, 3.6, 0.2, ""),
         (6.5, 0.75, 4.4, 0.2, "")]
for x1, y1, x2, y2, label in edges:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, fontsize=8, color="black")

ax.set_title("LangGraph: classify → (easy | hard) → END")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import matplotlib.patches as mpatches`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> LangGraph agents are evaluated by:
> 
> - **Path accuracy** — did the agent take the expected path through the graph?
> - **Final answer accuracy** — same as any agent.
> - **Step efficiency** — did it skip unnecessary nodes?
> - **Recovery** — when a node fails, does the graph route around it?

> **What this does:** Runs our TinyGraph on the eval set and reports path
> accuracy: did the agent take the expected route?
> 
> **Why it matters:** path accuracy is the unique metric for graph-based
> agents. A ReAct agent has no "path" — it just takes steps. A LangGraph
> agent has an explicit path, and you can measure whether the routing
> decisions were correct.
> 
> **What to watch for:** we just check the final answer here. To check the
> actual path, you'd record which nodes were visited and compare to the
> expected path.


### Code cell #5
```python
path_correct = 0
for qa in graph_eval:
    g = TinyGraph({"question": qa["q"]})
    g.add_node("classify", classify)
    g.add_node("answer_easy", answer_easy)
    g.add_node("answer_hard", answer_hard)
    g.add_edge("classify", route)
    g.set_entry("classify")
    out = g.run()
    # The path is implicit in the final answer's tag.
    took_easy = "(easy)" in out.get("answer", "")
    expected_easy = qa["expected_path"][-1] == "answer_easy"
    ok = took_easy == expected_easy
    path_correct += int(ok)
    print(f"  Q: {qa['q'][:40]:42s}  ok={ok}")

print(f"\npath accuracy: {path_correct}/{len(graph_eval)}")
```

**What this code does:**

- **Headline:** `path_correct = 0`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a "validate" node that checks the answer and routes back to "answer_hard" if invalid.
> 2. Add a "summarise" node that runs every 5 steps to compress the state.
> 3. Use LangGraph's checkpointing (`MemorySaver`) to resume a long-running agent.
> 4. Build a multi-agent graph: one agent researches, another writes, a third reviews.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a custom state reducer that accumulates a list (e.g. for message history).
> 2. Build a self-correcting agent: a "critic" node that can route back to "answer".
> 3. Implement parallel branches with `langgraph` and combine results.
> 4. Build a graph that supports human-in-the-loop checkpoints (pause + resume).
> 5. Implement a graph-based RAG system: route queries by type, retrieve, generate.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What problem does LangGraph solve that LangChain's AgentExecutor doesn't?
> 2. What is a conditional edge, and when would you use one?
> 3. What is a state reducer, and why does it matter for parallel branches?
> 4. How would you implement a multi-agent system in LangGraph?
> 5. What is checkpointing, and when is it useful?
> 6. How would you debug a LangGraph agent that takes the wrong branch?
> 7. What is the difference between a state schema and a state reducer?
> 8. When would you NOT use LangGraph?
