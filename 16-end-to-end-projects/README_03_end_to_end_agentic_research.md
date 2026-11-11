# Code Walkthrough — `03_end_to_end_agentic_research.ipynb`

> Module: **16-end-to-end-projects** | Notebook: **03_end_to_end_agentic_research.ipynb**
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

> # 03 — End-to-End Agentic: Research Assistant
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
> ## 1. Concept Overview — End-to-end agentic: research assistant

> This notebook builds a real agent: a research assistant that can search,
> summarise, and write. It uses the patterns from module 11 (ReAct, tool
> calling) plus a LangGraph-style control flow.
> 
> **Beginner track:** follow the 4 steps: define tools → build the agent
> loop → test → evaluate.
> **Advanced track:** also add multi-agent patterns (a critic that reviews
> the researcher's drafts).

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> The research agent is a ReAct loop (module 11) with three tools:
> `search`, `summarise`, `write`. Each tool is a function the agent can call.
> The agent picks the next tool based on the current state (query + history).
> 
> State transitions: query → search → summarise → write → review → final.
> Stop condition: the agent emits `Final Answer:` or hits max steps.

> ---
> 
> ## 3. Implementation from Scratch — Step 1 — Define the tools

> **What this does:** Defines three tools for the research agent: `search`
> (returns mock search results), `summarise` (mock summarises a list of
> texts), `write` (mock writes a short article).
> 
> **Why it matters:** the tool set *is* the agent's capability. A research
> agent without a search tool can't research; one without a write tool can't
> produce a report.
> 
> **What to watch for:** real tools call real APIs (web search, document DB,
> LLM). Our mocks are deterministic so we can test the loop logic.


### Code cell #1
```python
TOOLS = {}

def register_tool(name):
    def deco(fn):
        TOOLS[name] = fn
        return fn
    return deco

@register_tool("search")
def search(query):
    """Mock search: returns canned results per topic keyword."""
    if "transformer" in query.lower():
        return ["Vaswani et al. 2017 introduced the transformer.",
                "Transformers use self-attention, not recurrence."]
    if "rag" in query.lower():
        return ["RAG combines retrieval and generation.",
                "RAG grounds LLMs in external knowledge."]
    return ["No relevant results found."]

@register_tool("summarise")
def summarise(texts):
    """Mock summarise: returns first 80 chars of concatenated input."""
    combined = " ".join(texts)
    return combined[:80] + "..."

@register_tool("write")
def write(topic, summary):
    """Mock write: returns a short article based on topic + summary."""
    return f"# {topic}\n\nBased on research: {summary}"

print(f"registered tools: {list(TOOLS.keys())}")
print(f"search demo: {search('transformer architecture')[:1]}")
print(f"summarise demo: {summarise(['a b c d e f g h'])}")
print(f"write demo: {write('Transformers', 'self-attention based')[:50]}")
```

**What this code does:**

- **Headline:** `TOOLS = {}`
- **Defines:** def `register_tool`; def `search` — Mock search: returns canned results per topic keyword.; def `summarise` — Mock summarise: returns first 80 chars of concatenated input.; def `write` — Mock write: returns a short article based on topic + summary.
- **Notable calls:** `RAG`

> **Takeaway:** Tools are the agent's hands. The skill of agent engineering
> is choosing the right tools — too few and the agent can't do the task;
> too many and it can't choose.

> ---
> 
> ## 4. Library Implementation — Step 2 — Build the agent loop

> **What this does:** Implements the research agent as a state machine:
> query → search → summarise → write → final. Each step calls a tool and
> stores the result in the shared state. The loop runs until it reaches the
> final state.
> 
> **Why it matters:** this is the simplest agent that produces a multi-step
> output. More complex agents use an LLM to pick the next step; we hardcode
> the sequence for clarity.
> 
> **What to watch for:** real agents use an LLM to decide the next tool —
> the state machine pattern is a simplification. But the structure (state +
> step functions + loop) is the same.


### Code cell #2
```python
class ResearchAgent:
    def __init__(self, topic, max_steps=10):
        self.state = {"topic": topic, "history": [], "step": 0}
        self.max_steps = max_steps

    def step(self):
        # Hardcoded sequence: search -> summarise -> write -> done.
        step = self.state["step"]
        if step == 0:
            results = search(self.state["topic"])
            self.state["search_results"] = results
            self.state["history"].append({"step": "search", "output": results})
        elif step == 1:
            summary = summarise(self.state["search_results"])
            self.state["summary"] = summary
            self.state["history"].append({"step": "summarise", "output": summary})
        elif step == 2:
            article = write(self.state["topic"], self.state["summary"])
            self.state["article"] = article
            self.state["history"].append({"step": "write", "output": article})
        elif step == 3:
            self.state["final"] = self.state["article"]
            return True   # done
        self.state["step"] += 1
        return False

    def run(self):
        for _ in range(self.max_steps):
            done = self.step()
            if done: break
        return self.state

# Run on a topic.
agent = ResearchAgent("transformer architecture")
final_state = agent.run()
print(f"completed in {final_state['step']} steps")
print(f"final article:\n{final_state.get('final', '')[:200]}")
```

**What this code does:**

- **Headline:** `class ResearchAgent:`
- **Defines:** class `ResearchAgent`

> **Takeaway:** The agent loop is `state + step_fn + repeat`. The complexity
> in real agents is in the *decision* of which step to take next — we
> hardcoded it for the demo.

> ---
> 
> ## 5. Dataset Exploration — Step 3 — Test on multiple topics

> **What this does:** Runs the agent on 3 topics and reports the steps
> taken + final article. Each topic exercises the same tool sequence but
> with different content.
> 
> **Why it matters:** testing an agent on multiple inputs is the only way
> to know if it generalises. Single-input **What to watch for:** real agent eval uses 10-100 inputs and reports
> success rate. Our 3-input test is a smoke test, not a benchmark.


### Code cell #3
```python
topics = [
    "transformer architecture",
    "retrieval-augmented generation",
    "machine learning fundamentals",
]
for topic in topics:
    agent = ResearchAgent(topic)
    final_state = agent.run()
    print(f"\n=== topic: {topic} ===")
    print(f"steps taken: {final_state['step']}")
    print(f"search results: {final_state.get('search_results', [])}")
    print(f"final article (first 100 chars): {final_state.get('final', '')[:100]}...")
```

**What this code does:**

- **Headline:** `topics = [`

> ---
> 
> ## 6. Visualization — Step 4 — Evaluation

> **What this does:** A simple agent evaluator: did the agent produce a
> non-empty final article containing the topic keyword?
> 
> **Why it matters:** agent evaluation is hard because outputs are open-
> ended. A eval uses LLM-as-judge.
> 
> **What to watch for:** real eval has more dimensions — accuracy,
> completeness, format compliance, latency, cost. We check only the most
> basic: non-empty + on-topic.


### Code cell #4
```python
def evaluate_agent(topic, final_state):
    article = final_state.get("final", "")
    if not article:
        return False, "no article produced"
    topic_words = set(topic.lower().split())
    article_words = set(article.lower().split())
    overlap = len(topic_words & article_words) / max(1, len(topic_words))
    if overlap < 0.3:
        return False, f"low topic overlap ({overlap:.0%})"
    return True, f"on-topic ({overlap:.0%} overlap, {len(article)} chars)"

for topic in topics:
    agent = ResearchAgent(topic)
    final_state = agent.run()
    ok, reason = evaluate_agent(topic, final_state)
    print(f"  topic: {topic:40s} ok={ok}  {reason}")
```

**What this code does:**

- **Headline:** `def evaluate_agent(topic, final_state):`
- **Defines:** def `evaluate_agent`

> ---
> 
> ## 7. Model Evaluation — Step 5 — Serving with FastAPI

> **What this does:** Wraps the research agent in a FastAPI endpoint:
> `POST /research` accepts a topic and returns the final article.
> 
> **Why it matters:** an agent in a notebook is a demo; an agent in a
> FastAPI service is a product. The endpoint pattern is the same as the
> ML and RAG notebooks — the only difference is the agent takes longer.
> 
> **What to watch for:** long-running agents (>10s) should run as a
> background task. Use `BackgroundTasks` or a job queue (Celery, RQ). The
> HTTP request returns a job ID; the client polls a `/status/<job_id>`
> endpoint.


### Code cell #5
```python
from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)

class ResearchResponse(BaseModel):
    topic: str
    article: str
    steps: int

# In production:
# @app.post("/research", response_model=ResearchResponse)
# def research_endpoint(req: ResearchRequest):
#     agent = ResearchAgent(req.topic)
#     final = agent.run()
#     return ResearchResponse(topic=req.topic, article=final.get("final", ""),
#                            steps=final["step"])

# Mock the endpoint call.
req = ResearchRequest(topic="transformer architecture")
agent = ResearchAgent(req.topic)
final = agent.run()
resp = ResearchResponse(topic=req.topic, article=final.get("final", ""),
                        steps=final["step"])
print(f"input topic: {resp.topic}")
print(f"steps: {resp.steps}")
print(f"article (first 100 chars): {resp.article[:100]}...")
```

**What this code does:**

- **Headline:** `from pydantic import BaseModel, Field`
- **Defines:** class `ResearchRequest`; class `ResearchResponse`
- **Key imports:** `from pydantic import BaseModel`

> ---
> 
> ## 8. Experiments — Step 6 — Production monitoring

> **What this does:** A simple monitoring shim: time each agent run, log
> the topic, steps, and duration. This is the *minimum* monitoring for any
> production agent.
> 
> **Why it matters:** agents are stochastic — the same topic may produce
> different outputs across runs. Without monitoring, you can't tell if a
> user complaint is a real bug or a bad roll of the dice.
> 
> **What to watch for:** real agent monitoring uses LangSmith or Langfuse
> to capture every tool call, every LLM response, and every intermediate
> state. Our shim is the minimum viable version.


### Code cell #6
```python
import time
import json
from pathlib import Path

monitor_log = []

def run_with_monitoring(topic):
    t0 = time.perf_counter()
    agent = ResearchAgent(topic)
    final = agent.run()
    duration = time.perf_counter() - t0
    log_entry = {
        "topic": topic, "steps": final["step"], "duration_s": duration,
        "article_length": len(final.get("final", "")),
    }
    monitor_log.append(log_entry)
    return final, log_entry

# Run a few times.
for topic in topics:
    run_with_monitoring(topic)

# Print the monitoring log.
print("monitoring log:")
for entry in monitor_log:
    print(f"  topic={entry['topic'][:30]:32s} steps={entry['steps']} "
          f"dur={entry['duration_s']*1000:.1f}ms len={entry['article_length']}")
```

**What this code does:**

- **Headline:** `import time`
- **Defines:** def `run_with_monitoring`
- **Key imports:** `import time`, `import json`, `from pathlib import Path`

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Replace the hardcoded state machine with an LLM-driven decision (ReAct).
> 2. Add a `critic` tool that reviews the draft and asks for revisions.
> 3. Use LangGraph to express the same workflow as a graph.
> 4. Add memory: the agent remembers past queries and reuses their summaries.
> 5. Deploy with a job queue (Celery) for long-running research tasks.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Walk through the architecture of a production agent.
> 2. How do you choose the tool set for an agent?
> 3. How do you evaluate an agent?
> 4. What are the common failure modes of agents?
> 5. How do you handle long-running agents in a web service?
> 6. What is the role of monitoring in agent production?
> 7. How do you prevent an agent from looping indefinitely?
> 8. How do you handle multi-tenant agent isolation?
