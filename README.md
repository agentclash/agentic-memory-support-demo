# agentic-memory-support-demo

A small chatbot demo that shows why `agentic-memory` matters.

This repo pairs a support-style chatbot with a deterministic benchmark harness.
The benchmark compares the exact same bot in two modes:

- `baseline`: no long-term memory
- `memory_enabled`: semantic, episodic, and procedural memory via the `agentic-memory` SDK

The point is not "chatbot vibes". The point is to show measurable behavior changes:

- remembering user profile facts across turns
- not asking users to repeat troubleshooting steps
- retrieving the right procedure for the current issue

## What This Demonstrates

Most chatbot demos look better because the prompt got longer or the examples got hand-picked.

This repo tries to isolate the contribution of memory itself:

- the same deterministic LLM is used in both benchmark modes
- the only difference is whether the bot can write and retrieve memories
- scoring is exact substring matching, not an LLM judge

## Deterministic Benchmark

The benchmark includes 12 synthetic support scenarios:

- 4 current-turn control cases
- 4 cross-turn profile recall cases
- 2 troubleshooting continuity cases
- 2 procedure retrieval cases

Expected benchmark result:

| Metric | Baseline | Memory Enabled |
|---|---:|---:|
| Overall accuracy | 33.3% | 100.0% |
| Current-turn controls | 100.0% | 100.0% |
| Profile recall | 0.0% | 100.0% |
| Troubleshooting continuity | 0.0% | 100.0% |
| Procedure retrieval | 0.0% | 100.0% |

## Quickstart

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### 2. Run the benchmark

```bash
memory-support-benchmark
```

### 3. Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

If `OPENAI_API_KEY` is set, the app uses a real OpenAI-compatible model.
If not, it falls back to a deterministic responder so the demo still works offline.

## Why The Demo Is Structured This Way

This repo is intentionally opinionated:

- semantic memory stores durable user facts like plan, timezone, and preferences
- episodic memory stores troubleshooting history and prior conversation events
- procedural memory stores reusable support runbooks

That makes the value proposition legible in one sentence:

> the bot remembers who the user is, what already happened, and what usually works

## Project Layout

```text
agentic-memory-support-demo/
├── evals/
│   └── run_benchmark.py
├── src/
│   └── agentic_memory_support_demo/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── chatbot.py
│       ├── deterministic.py
│       └── llm.py
├── tests/
│   └── test_benchmark.py
├── pyproject.toml
└── streamlit_app.py
```
