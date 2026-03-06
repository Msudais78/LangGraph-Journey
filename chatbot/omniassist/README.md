# 🤖 OmniAssist — AI-Powered Multi-Agent Conversational Platform

Built on **LangGraph v1.0+**, powered by **Groq** (free LLMs).

## Features

- **Multi-Agent Architecture:** Supervisor routes to 6 specialist agents
  - Chat, Research, Writing, Task Management, Code Execution, Data Analysis
- **Both APIs:** StateGraph (Graph API) + `@entrypoint`/`@task` (Functional API)
- **Human-in-the-Loop:** `interrupt()` / `Command(resume=...)` pattern
- **Memory:** Short-term (context trimming), long-term (InMemoryStore)
- **Streaming:** Progress events, thinking indicators, agent handoff events
- **Checkpointing:** InMemory (dev), SQLite/PostgreSQL (prod)
- **Frontend:** Dark-themed chat UI with HITL approval bar

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set your Groq API key
# Edit .env and set GROQ_API_KEY=your_key_here

# 3. Run CLI chat
python scripts/cli_chat.py
```

## Available Groq Models

| Model                     | Use case               |
| ------------------------- | ---------------------- |
| `llama-3.3-70b-versatile` | Default — best quality |
| `llama-3.1-8b-instant`    | Fast responses         |
| `mixtral-8x7b-32768`      | Long context           |
| `gemma2-9b-it`            | Lightweight            |

Switch models in the CLI:

```
switch fast   # uses llama-3.1-8b-instant
switch default  # uses llama-3.3-70b-versatile
```

## Run Tests

```bash
python -m pytest tests/ -v --tb=short
```

## Architecture

```
User Input
    ↓
input_handler       → validate input
    ↓
supervisor          → classify intent via Groq → Command(goto=agent)
    ↓
[chat_agent | research | writing | task_mgmt | code_exec | data_analysis]
    ↓
memory_manager      → store facts in InMemoryStore
    ↓
output_handler      → format response
    ↓
Response
```

## Project Structure

```
src/
├── state/          # TypedDict schemas + reducers
├── graphs/         # StateGraph subgraphs (chat, research, writing, etc.)
├── workflows/      # Functional API (@entrypoint/@task) code pipeline
├── nodes/          # Supervisor, input/output handlers, memory manager
├── tools/          # 10 tools (calculator, web search, code runner, etc.)
├── memory/         # Short-term & long-term memory
├── agents/         # Supervisor/swarm/bigtool setups
├── config/         # Settings (Groq key), models (ChatGroq), prompts
└── utils/          # Streaming, validation, formatters, checkpointing
scripts/
├── cli_chat.py     # Rich CLI interface
└── benchmark.py    # Performance benchmarking
frontend/           # HTML/CSS/JS chat UI (for LangGraph Platform)
```

## Deployment (LangGraph Platform)

```bash
langgraph dev   # Local development with LangGraph Studio
langgraph up    # Production deployment
```
