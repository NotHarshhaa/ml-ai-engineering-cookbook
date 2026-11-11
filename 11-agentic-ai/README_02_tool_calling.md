# Code Walkthrough — `02_tool_calling.ipynb`

> Module: **11-agentic-ai** | Notebook: **02_tool_calling.ipynb**
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

> # 02 — Tool Calling
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
> ## 1. Concept Overview — Tool calling — agents that use functions

> Tool calling (also called *function calling*) is the mechanism by which an
> LLM invokes external functions. The model is given a schema describing the
> available functions; when it decides a function should be called, it emits
> structured arguments that the framework parses and executes.
> 
> **Beginner track:** learn the JSON-schema tool definition format. That's
> what every framework (LangChain, OpenAI, Anthropic) uses under the hood.
> **Advanced track:** also learn parallel tool calls, multi-turn tool
> conversations, and how to enforce output schema with `pydantic`.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Tool calling extends the LLM's output distribution: instead of just tokens,
> the model can emit structured JSON matching a provided schema. The LLM is
> fine-tuned (or prompted) to emit a JSON object $\{\text{name}, \text{args}\}$
> when the prompt suggests a tool call is appropriate.
> 
> The framework parses the JSON, dispatches to the actual function, appends
> the result to the conversation, and re-invokes the LLM. Loop until no more
> tool calls.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a tiny tool-calling loop with a mock LLM
> that emits JSON-formatted tool calls. The framework parses the JSON,
> dispatches to the right function, and feeds the result back.
> 
> **Why it matters:** this is exactly what OpenAI's `function_call` API does,
> and what every tool-calling framework implements. Seeing it in 50 lines
> makes it unmagical.
> 
> **What to watch for:** the LLM's output is *text* (a JSON string), not a
> structured object. The framework parses it; if parsing fails, the loop
> fails. Always validate tool-call JSON.


### Code cell #1
```python
import json

# Tool registry: name -> (function, schema).
def get_weather(city):
    # Mock weather function.
    return {"city": city, "temperature_c": 22, "condition": "sunny"}

def get_population(country):
    # Mock population function.
    pops = {"France": 67_000_000, "Germany": 83_000_000, "Japan": 125_000_000}
    return {"country": country, "population": pops.get(country, "unknown")}

TOOLS = {
    "get_weather": {"fn": get_weather, "schema": {"city": "str"}},
    "get_population": {"fn": get_population, "schema": {"country": "str"}},
}

# Mock LLM that decides which tool to call based on the user query.
def mock_tool_calling_llm(query):
    if "weather" in query.lower():
        return json.dumps({"name": "get_weather", "args": {"city": "Paris"}})
    if "population" in query.lower():
        return json.dumps({"name": "get_population", "args": {"country": "France"}})
    return json.dumps({"name": "_final", "args": {"answer": "I don't know."}})

def tool_calling_agent(query, max_steps=3):
    conversation = [{"role": "user", "content": query}]
    for step in range(max_steps):
        # LLM emits a tool call (as JSON text).
        llm_output = mock_tool_calling_llm(query)
        try:
            call = json.loads(llm_output)
        except json.JSONDecodeError:
            return f"error: invalid tool-call JSON: {llm_output}"
        if call["name"] == "_final":
            return call["args"]["answer"]
        if call["name"] not in TOOLS:
            return f"error: unknown tool {call['name']}"
        # Dispatch and observe.
        result = TOOLS[call["name"]]["fn"](**call["args"])
        conversation.append({"role": "tool", "name": call["name"], "content": json.dumps(result)})
        # In a real agent the LLM would see the result and decide next; we
        # shortcut by returning the result for the demo.
        return f"Tool result: {result}"
    return "max steps reached"

print(tool_calling_agent("What's the weather in Paris?"))
print(tool_calling_agent("What's the population of France?"))
```

**What this code does:**

- **Headline:** `import json`
- **Defines:** def `get_weather`; def `get_population`; def `mock_tool_calling_llm`; def `tool_calling_agent`
- **Key imports:** `import json`

> **Takeaway:** Tool calling = JSON schema describing tools + LLM emitting JSON
> calls + framework parsing and dispatching. The hard part is error handling
> and parsing edge cases, not the loop itself.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Shows the OpenAI function-calling API pattern with the
> exact JSON-schema format. We don't call the live API (no key required) —
> the code is what you'd ship to production.
> 
> **Why it matters:** OpenAI's function-calling is the de-facto schema format.
> Anthropic, Mistral, and Google all support essentially the same structure.
> Learn it once, use it everywhere.
> 
> **What to watch for:** `parameters: {"type": "object", "properties": ...}`
> is JSON-schema. You can express required vs optional, types, enums, and
> nested structures. Pydantic models auto-convert to this.


### Code cell #2
```python
# The standard tool schema format used by OpenAI/Anthropic/Mistral.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population of a country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "Country name."},
                },
                "required": ["country"],
            },
        },
    },
]

print("Tool schema format (production OpenAI-style):")
import json
print(json.dumps(tools_schema, indent=2))
```

**What this code does:**

- **Headline:** `tools_schema = [`
- **Key imports:** `import json`

> **What this does:** Demonstrates `pydantic` for tool-argument validation —
> the standard way to define tool schemas in Python agent frameworks.
> 
> **Why it matters:** pydantic models give you (a) JSON-schema for free,
> (b) runtime validation of the LLM's arguments, and (c) IDE autocomplete
> for the function call. Always use pydantic over hand-written schemas.
> 
> **What to watch for:** `Field(..., description=...)` becomes the LLM-visible
> description. Good descriptions are the single biggest lever on tool-call
> accuracy — the LLM picks tools based on them.


### Code cell #3
```python
from pydantic import BaseModel, Field

class WeatherArgs(BaseModel):
    """Arguments for the get_weather tool."""
    city: str = Field(..., description="City name, e.g. 'Paris'.")
    units: str = Field("celsius", description="'celsius' or 'fahrenheit'.")

class PopulationArgs(BaseModel):
    """Arguments for the get_population tool."""
    country: str = Field(..., description="Country name, e.g. 'France'.")

# pydantic auto-generates the JSON schema.
print("WeatherArgs schema:")
print(WeatherArgs.model_json_schema())

# Runtime validation: parses the LLM's JSON call.
try:
    args = WeatherArgs.model_validate({"city": "Paris", "units": "celsius"})
    print(f"\nvalidated: {args}")
except Exception as e:
    print(f"validation error: {e}")
```

**What this code does:**

- **Headline:** `from pydantic import BaseModel, Field`
- **Defines:** class `WeatherArgs` — Arguments for the get_weather tool.; class `PopulationArgs` — Arguments for the get_population tool.
- **Key imports:** `from pydantic import BaseModel`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Builds a small set of tool-using tasks: 5 multi-step
> queries that require calling multiple tools in sequence.
> 
> **Why it matters:** multi-step tool use is what distinguishes an agent from
> a single tool call. The benchmark should require *chaining* — e.g. call
> get_population, then call calculator on the result.
> 
> **What to watch for:** the gold trace specifies which tools should be called
> in what order. Real agent eval compares the agent's trace to the gold
> trace.


### Code cell #4
```python
tool_eval = [
    {"q": "Weather in Paris and Tokyo", "gold_tools": ["get_weather", "get_weather"]},
    {"q": "Population of France and Germany", "gold_tools": ["get_population", "get_population"]},
    {"q": "Compare temperature and population of Japan",
     "gold_tools": ["get_weather", "get_population"]},
    {"q": "Population of France times 3", "gold_tools": ["get_population", "calculator"]},
]
print(f"tool eval size: {len(tool_eval)}")
for qa in tool_eval:
    print(f"  Q: {qa['q']}")
    print(f"    gold trace: {qa['gold_tools']}")
```

**What this code does:**

- **Headline:** `tool_eval = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises an agent's tool-call trace as a flow chart:
> user → tool → result → tool → result → final answer. This is the standard
> way to present agent traces in demos and reports.
> 
> **Why it matters:** visualising the trace is the most direct way to debug
> agent behaviour. If your agent chose the wrong tool, the trace shows it
> immediately.
> 
> **What to watch for:** we use matplotlib boxes and arrows. Production tools
> like LangSmith provide interactive HTML trace viewers.


### Code cell #5
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

# Boxes: (label, x, y, color)
boxes = [
    ("User query",        1, 2, "#3498db"),
    ("LLM (step 1)",      3, 2, "#9b59b6"),
    ("get_population",    5, 3, "#27ae60"),
    ("get_weather",       5, 1, "#27ae60"),
    ("LLM (step 2)",      7, 2, "#9b59b6"),
    ("Final answer",      9, 2, "#e74c3c"),
]
for label, x, y, color in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="white", alpha=0.8))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")

# Arrows.
arrows = [(1.6, 2, 2.4, 2), (3.6, 2.2, 4.4, 2.8), (3.6, 1.8, 4.4, 1.2),
         (5.6, 2.8, 6.4, 2.2), (5.6, 1.2, 6.4, 1.8), (7.6, 2, 8.4, 2)]
for x1, y1, x2, y2 in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

ax.set_title("Agent tool-call trace: parallel tool calls then final answer")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import matplotlib.patches as mpatches`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Tool-calling agents are evaluated by:
> 
> - **Tool selection accuracy** — did the LLM pick the right tool?
> - **Argument accuracy** — were the arguments parsed correctly?
> - **Trace accuracy** — did the agent call tools in the right order?
> - **Final answer accuracy** — did the chain produce the right result?

> **What this does:** A tiny tool-call evaluator that checks argument
> parsing accuracy. We feed valid and invalid JSON to the pydantic model and
> count how often it correctly validates or rejects.
> 
> **Why it matters:** argument parsing is the single most common failure
> mode for tool-calling agents. If the LLM emits `{"city": 123}` (number
> instead of string), pydantic should catch it. Without that, the tool
> function crashes.
> 
> **What to watch for:** pydantic raises `ValidationError` on bad inputs;
> the agent framework should catch this and feed the error back to the LLM
> as an observation, so the LLM can retry with corrected arguments.


### Code cell #6
```python
# Test argument validation against good and bad inputs.
test_cases = [
    ({"city": "Paris", "units": "celsius"}, True),
    ({"city": "Paris", "units": "fahrenheit"}, True),
    ({"city": "Paris"}, True),                       # units has default
    ({"city": 123}, False),                          # city must be str
    ({"units": "celsius"}, False),                   # missing required city
    ({"city": "Paris", "units": "kelvin"}, True),    # units has no enum constraint
]

passed = 0
for args, expected_valid in test_cases:
    try:
        WeatherArgs.model_validate(args)
        valid = True
    except Exception:
        valid = False
    ok = valid == expected_valid
    '    print(f"  args={args}  expected_valid={expected_valid}  actual={valid}  ok_or_FAIL={ok}")\n'    passed += int(ok)
print(f"\nvalidation accuracy: {passed}/{len(test_cases)}")
```

**What this code does:**

- **Headline:** `test_cases = [`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Add a `calculator` tool and let the agent chain population × 3.
> 2. Try parallel tool calls: have the LLM emit multiple tool calls in one step.
> 3. Add an `error` channel — let the LLM see when a tool fails and retry.
> 4. Use `langchain.tools.Tool` to define the same tools, then run with a real LLM.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a `@tool` decorator that auto-generates the JSON schema from a function's type hints.
> 2. Build a tool that lets the agent query a SQL database safely (read-only).
> 3. Implement parallel tool execution with `asyncio.gather`.
> 4. Build a tool-retrieval system: with 1000 tools, embed tool descriptions and retrieve the top-5 for each query.
> 5. Implement a tool-calling eval suite that scores trace accuracy.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between prompting an LLM to "act" and proper tool calling?
> 2. What is the role of JSON schema in tool calling?
> 3. Why use pydantic for tool arguments?
> 4. How do you handle invalid tool arguments produced by the LLM?
> 5. What are parallel tool calls, and when do they help?
> 6. How would you support 1000+ tools when the LLM context is limited?
> 7. What is the difference between OpenAI function calling and Anthropic tool use?
> 8. How would you evaluate a tool-calling agent?
