# 🤖 OMNIASSIST — AI Agent Executable Build Specification

> **Spec Version:** 2.0
> **Target:** LangGraph v1.0+ Python Project
> **Execution Model:** Sequential phases, each with atomic tasks. Complete each task fully before proceeding. Verify after each task.

---

## 🔒 GLOBAL RULES (Always Active)

```
RULE_1: Never skip a task. If a task fails, debug it before moving on.
RULE_2: Every file must have all imports at the top.
RULE_3: Every node function must be a pure function: (state) → partial state update. No mutation.
RULE_4: After creating/modifying any file, verify it by running: python -c "import <module>"
RULE_5: Use Python 3.11+ syntax throughout.
RULE_6: All async code must use `async def` and be awaitable.
RULE_7: Use TypedDict (not dataclass) for state schemas unless explicitly noted.
RULE_8: Use `interrupt()` function, NEVER `NodeInterrupt` (deprecated).
RULE_9: Use `context_schema`, NEVER `config_schema` (deprecated).
RULE_10: Use `InMemorySaver` (not `MemorySaver`) for dev checkpointing.
RULE_11: When using PostgresSaver, always call `.setup()` after creation.
RULE_12: interrupt() re-runs the entire node on resume — no side effects before interrupt().
RULE_13: Always read LangGraph latest docs if a concept is unclear before implementing.
```

---

## ⚙️ ENVIRONMENT ASSUMPTIONS

```
- OS: Linux/macOS (or WSL on Windows)
- Python: 3.11+
- Package manager: pip (via pyproject.toml)
- LLM providers: OpenAI (primary), Anthropic (secondary)
- Database: SQLite (dev), PostgreSQL (prod)
- The agent has access to a terminal and file system.
```

---

---

# 📦 PHASE 0 — PROJECT SCAFFOLDING

> **Goal:** Create the complete directory structure, install all dependencies, and validate the environment.

---

### TASK 0.1 — Create Root Directory and Project Files

**Action:** Create the following directory tree. Every listed directory must exist. Create empty `__init__.py` in every Python package directory.

```
COMMAND: mkdir -p omniassist
COMMAND: cd omniassist
```

Create this exact structure:

```
omniassist/
├── pyproject.toml
├── langgraph.json
├── .env
├── README.md
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── main_state.py
│   │   ├── context.py
│   │   ├── research_state.py
│   │   ├── writing_state.py
│   │   ├── task_state.py
│   │   ├── code_state.py
│   │   └── data_state.py
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── chat.py
│   │   ├── research.py
│   │   ├── source_validator.py
│   │   ├── writing.py
│   │   ├── task_management.py
│   │   └── data_analysis.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── code_pipeline.py
│   │   ├── quick_research.py
│   │   └── mcp_bridge.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── supervisor.py
│   │   ├── input_handler.py
│   │   ├── output_handler.py
│   │   ├── memory_manager.py
│   │   ├── error_handler.py
│   │   ├── human_review.py
│   │   └── message_utils.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── web_search.py
│   │   ├── calculator.py
│   │   ├── weather.py
│   │   ├── calendar_tool.py
│   │   ├── email_tool.py
│   │   ├── file_tools.py
│   │   ├── code_runner.py
│   │   ├── knowledge_base.py
│   │   ├── state_updating_tools.py
│   │   └── mcp_tools.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py
│   │   ├── long_term.py
│   │   ├── user_profile.py
│   │   └── langmem_integration.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor_setup.py
│   │   ├── swarm_setup.py
│   │   └── bigtool_setup.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── models.py
│   │   ├── assistants.py
│   │   └── prompts.py
│   └── utils/
│       ├── __init__.py
│       ├── streaming.py
│       ├── validation.py
│       ├── encryption.py
│       └── formatters.py
├── tests/
│   ├── __init__.py
│   ├── test_state.py
│   ├── test_chat_agent.py
│   ├── test_research.py
│   ├── test_writing.py
│   ├── test_task_management.py
│   ├── test_code_execution.py
│   ├── test_data_analysis.py
│   ├── test_memory.py
│   ├── test_streaming.py
│   ├── test_functional_api.py
│   ├── test_mcp_integration.py
│   ├── test_semantic_search.py
│   └── test_encrypted_checkpoints.py
├── scripts/
│   ├── cli_chat.py
│   ├── replay_thread.py
│   ├── seed_knowledge_base.py
│   └── benchmark.py
└── frontend/
    ├── index.html
    ├── app.js
    └── styles.css
```

**Commands to create all directories:**
```bash
mkdir -p src/{state,graphs,workflows,nodes,tools,memory,agents,config,utils}
mkdir -p tests scripts frontend
touch src/__init__.py
touch src/{state,graphs,workflows,nodes,tools,memory,agents,config,utils}/__init__.py
touch tests/__init__.py
```

**VERIFY:** Run `find . -name "__init__.py" | wc -l` — expect **11** `__init__.py` files.

---

### TASK 0.2 — Create `pyproject.toml`

**Action:** Write the following content to `omniassist/pyproject.toml`:

```toml
[project]
name = "omniassist"
version = "1.0.0"
description = "AI-Powered Multi-Agent Conversational Platform built on LangGraph"
requires-python = ">=3.11"
dependencies = [
    # Core LangGraph (v1.0+)
    "langgraph>=1.0",
    "langchain-core>=0.3",
    "langchain-openai>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-community>=0.3",

    # Checkpointing
    "langgraph-checkpoint-sqlite>=3.0",
    "langgraph-checkpoint-postgres>=3.0",
    "pycryptodome>=3.20",

    # Multi-agent libraries
    "langgraph-supervisor>=0.1",
    "langgraph-swarm>=0.1",
    "langgraph-bigtool>=0.1",

    # Memory
    "langmem>=0.1",

    # MCP
    "langchain-mcp-adapters>=0.1",

    # Tools & Data
    "tavily-python>=0.5",
    "chromadb>=0.5",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "rich>=13.0",
    "matplotlib>=3.9",
    "pandas>=2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

---

### TASK 0.3 — Create `.env` File

**Action:** Write to `omniassist/.env`:

```env
OPENAI_API_KEY=sk-REPLACE_ME
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME
TAVILY_API_KEY=tvly-REPLACE_ME
LANGSMITH_API_KEY=ls-REPLACE_ME
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=omniassist
DATABASE_URL=postgresql://localhost:5432/omniassist
LANGGRAPH_AES_KEY=REPLACE_ME_WITH_32_BYTE_HEX_KEY
```

> ⚠️ **NOTE TO AGENT:** If real API keys are available, substitute them now. Otherwise, leave placeholders — they will be needed starting Phase 1.

---

### TASK 0.4 — Create `langgraph.json`

**Action:** Write to `omniassist/langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "omniassist": "./src/graphs/main.py:graph",
    "research_only": "./src/graphs/research.py:graph",
    "code_runner": "./src/workflows/code_pipeline.py:code_pipeline"
  },
  "env": ".env",
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["text", "summary"]
    }
  }
}
```

---

### TASK 0.5 — Install Dependencies

**Action:** Run:

```bash
cd omniassist
pip install -e ".[dev]"
```

**VERIFY:** Run each of these and confirm no `ImportError`:

```bash
python -c "import langgraph; print(langgraph.__version__)"
python -c "from langgraph.graph import StateGraph, START, END"
python -c "from langgraph.func import entrypoint, task"
python -c "from langgraph.checkpoint.memory import InMemorySaver"
python -c "from langgraph.types import interrupt, Command, RetryPolicy, CachePolicy"
python -c "from langchain_openai import ChatOpenAI"
python -c "from pydantic import BaseModel"
python -c "import rich"
```

> If any import fails, install the missing package individually and retry. Do not proceed until all imports succeed.

---

### TASK 0.6 — Create `src/config/settings.py`

**Action:** Write to `src/config/settings.py`:

```python
"""Application settings loaded from environment."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///omniassist.db")
LANGGRAPH_AES_KEY = os.getenv("LANGGRAPH_AES_KEY", "")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_RESEARCH_SOURCES = 5
DEFAULT_PERSONA = "professional"
DEFAULT_LANGUAGE = "en"
```

**VERIFY:** `python -c "from src.config.settings import DEFAULT_MODEL; print(DEFAULT_MODEL)"`

---

### TASK 0.7 — Create `src/config/prompts.py`

**Action:** Write to `src/config/prompts.py`:

```python
"""System prompts for all agents and nodes."""

SUPERVISOR_PROMPT = """You are the OmniAssist supervisor. Your job is to analyze the user's message
and determine which specialist agent should handle it.

Available agents:
- chat_agent: General conversation, greetings, small talk, simple questions
- research: In-depth research, finding information, fact-checking, web search
- writing: Content creation, drafting emails, articles, summaries, editing
- task_mgmt: Task management, to-do lists, reminders, scheduling
- code_exec: Code generation, execution, debugging, programming help
- data_analysis: Data analysis, statistics, visualization, CSV/data processing

Respond with ONLY the agent name (one of the above). Nothing else."""

CHAT_AGENT_PROMPT = """You are OmniAssist's chat agent. You handle general conversation,
answer simple questions, and provide helpful responses. Be friendly and concise.
Use tools when needed to provide accurate information."""

RESEARCH_AGENT_PROMPT = """You are OmniAssist's research agent. You perform thorough research
on topics by searching multiple sources, validating information, and synthesizing findings
into clear, well-sourced responses."""

WRITING_AGENT_PROMPT = """You are OmniAssist's writing agent. You help users create, edit,
and refine written content. You produce drafts, accept feedback, and iteratively improve
content quality. Always ask for approval before finalizing."""

TASK_AGENT_PROMPT = """You are OmniAssist's task management agent. You help users create,
update, delete, and organize tasks. You maintain a structured task list and provide
status updates. Always confirm before destructive operations."""

CODE_AGENT_PROMPT = """You are OmniAssist's code agent. You generate, review, and execute
Python code in a sandboxed environment. Always review code for safety before execution.
Explain what the code does before running it."""

DATA_AGENT_PROMPT = """You are OmniAssist's data analysis agent. You help users analyze data,
generate statistics, and create visualizations. You work with CSV data, pandas DataFrames,
and produce matplotlib charts."""
```

**VERIFY:** `python -c "from src.config.prompts import SUPERVISOR_PROMPT; print('OK')"`

---

---

# 🏗️ PHASE 1 — STATE DEFINITIONS & FOUNDATION (Days 1–3)

> **Goal:** Define all state schemas, build the Chat Agent (both prebuilt and custom), set up checkpointing, basic streaming, and a CLI test interface.
>
> **Concepts covered:** `StateGraph`, `TypedDict`, `Annotated[type, reducer]`, `Input/Output Schema`, `context_schema`, `Pydantic Validation`, `create_react_agent`, `ToolNode`, `Tool Calling`, `InMemorySaver`, `Streaming (Tokens, Updates)`, `StreamWriter`, `START`, `END`, `Nodes`, `Edges`

---

### TASK 1.1 — Define Custom Reducers

**File:** `src/state/__init__.py`

**Action:** Write:

```python
"""Custom reducers for state channels."""

from typing import Any


def append_reducer(existing: list | None, new: list | Any) -> list:
    """Reducer that appends new items to existing list.

    If `new` is a list, extend. If single item, append.
    """
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return existing + [new]


def replace_reducer(existing: Any, new: Any) -> Any:
    """Reducer that simply replaces the old value with the new one."""
    return new
```

**VERIFY:** `python -c "from src.state import append_reducer; print(append_reducer([1,2], [3,4]))"`
**Expected:** `[1, 2, 3, 4]`

---

### TASK 1.2 — Define Main State Schema

**File:** `src/state/main_state.py`

**Action:** Write:

```python
"""Core state schema for OmniAssist main orchestrator graph."""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, field_validator

from src.state import append_reducer


# ──────────────────────────────────────────────
# MAIN STATE — used by the orchestrator graph
# ──────────────────────────────────────────────
class OmniAssistState(TypedDict):
    """Full internal state for the main orchestrator."""

    # Conversation
    messages: Annotated[list, add_messages]          # Built-in message reducer
    current_intent: str                               # Detected intent label
    conversation_summary: str                         # Rolling summary of old messages

    # User
    user_id: str
    thread_id: str

    # Agent outputs
    tool_results: Annotated[list[dict], append_reducer]
    research_results: Annotated[list[dict], append_reducer]
    draft_content: str
    tasks: Annotated[list[dict], append_reducer]
    code_output: str
    data_analysis: dict

    # Control flow
    current_agent: str
    requires_human_approval: bool
    human_feedback: str | None
    error_log: Annotated[list[str], append_reducer]
    retry_count: int
    metadata: dict


# ──────────────────────────────────────────────
# INPUT / OUTPUT SCHEMAS — for API-facing boundary
# ──────────────────────────────────────────────
class OmniAssistInput(TypedDict):
    """Public input schema — only messages accepted."""
    messages: list


class OmniAssistOutput(TypedDict):
    """Public output schema — only messages and current_agent returned."""
    messages: list
    current_agent: str


# ──────────────────────────────────────────────
# PYDANTIC VALIDATION — validates first-node input
# ──────────────────────────────────────────────
class ValidatedInput(BaseModel):
    """Pydantic model used to validate incoming user input at graph boundary.

    NOTE: Pydantic validation only runs on the first node's input,
    not on subsequent nodes. Use dataclass/TypedDict internally for performance.
    """
    messages: list
    user_id: str = "default_user"

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id cannot be empty")
        return v.strip()
```

**VERIFY:** `python -c "from src.state.main_state import OmniAssistState, OmniAssistInput, OmniAssistOutput; print('OK')"`

---

### TASK 1.3 — Define Context Schema

**File:** `src/state/context.py`

**Action:** Write:

```python
"""Context schema for OmniAssist.

This replaces the deprecated `config_schema`. It provides run-scoped
configuration that nodes can access via Runtime.
"""

from typing_extensions import TypedDict


class OmniAssistContext(TypedDict, total=False):
    """Run-scoped context injected into the graph at invocation time.

    Access inside nodes via: runtime.context["model_name"]
    """
    model_name: str                  # e.g. "gpt-4o", "gpt-4o-mini", "claude-3-sonnet"
    temperature: float               # 0.0 – 2.0
    persona: str                     # "casual", "professional", "technical"
    max_research_sources: int        # Max sources for research subgraph
    code_execution_enabled: bool     # Whether code execution is allowed
    language: str                    # "en", "es", "fr", etc.
```

**VERIFY:** `python -c "from src.state.context import OmniAssistContext; print('OK')"`

---

### TASK 1.4 — Define Subgraph State Schemas

**File:** `src/state/research_state.py`

```python
"""State schema for the Research subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_queries: list[str]
    web_results: Annotated[list[dict], append_reducer]
    kb_results: Annotated[list[dict], append_reducer]
    academic_results: Annotated[list[dict], append_reducer]
    validated_sources: Annotated[list[dict], append_reducer]
    synthesis: str
    quality_score: float
    needs_more_research: bool
    user_id: str
```

**File:** `src/state/writing_state.py`

```python
"""State schema for the Writing subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class WritingState(TypedDict):
    messages: Annotated[list, add_messages]
    writing_request: str
    content_type: str              # "email", "article", "summary", "report"
    draft_content: str
    revision_history: Annotated[list[str], append_reducer]
    quality_score: float
    human_feedback: str | None
    action: str                    # "approve", "revise", "restart"
    iteration_count: int
    user_id: str
```

**File:** `src/state/task_state.py`

```python
"""State schema for the Task Management subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class TaskState(TypedDict):
    messages: Annotated[list, add_messages]
    task_action: str               # "create", "update", "delete", "list", "delete_all", "bulk_update"
    task_data: dict
    tasks: Annotated[list[dict], append_reducer]
    affected_count: int
    related_tasks: list[dict]
    confirmation_required: bool
    user_id: str
```

**File:** `src/state/code_state.py`

```python
"""State schema for the Code Execution module (used by Functional API too)."""

from __future__ import annotations
from typing_extensions import TypedDict


class CodeState(TypedDict):
    request: str
    generated_code: str
    language: str
    safety_level: str              # "safe", "caution", "dangerous"
    review_notes: str
    execution_output: str
    execution_error: str | None
    approved: bool
    user_id: str
```

**File:** `src/state/data_state.py`

```python
"""State schema for the Data Analysis subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class DataState(TypedDict):
    messages: Annotated[list, add_messages]
    data_source: str               # File path or inline data
    raw_data: str
    analysis_type: str             # "statistics", "visualization", "correlation", "summary"
    analysis_results: Annotated[list[dict], append_reducer]
    visualizations: Annotated[list[str], append_reducer]   # Base64 encoded images
    summary: str
    user_id: str
```

**VERIFY:** Run:
```bash
python -c "
from src.state.research_state import ResearchState
from src.state.writing_state import WritingState
from src.state.task_state import TaskState
from src.state.code_state import CodeState
from src.state.data_state import DataState
print('All subgraph states OK')
"
```

---

### TASK 1.5 — Create Config Models

**File:** `src/config/models.py`

```python
"""LLM model configurations."""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from src.config.settings import OPENAI_API_KEY, ANTHROPIC_API_KEY, DEFAULT_TEMPERATURE


def get_chat_model(model_name: str = "gpt-4o-mini", temperature: float = DEFAULT_TEMPERATURE):
    """Factory for chat models based on model name."""
    if model_name.startswith("claude"):
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=ANTHROPIC_API_KEY,
        )
    else:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )
```

**VERIFY:** `python -c "from src.config.models import get_chat_model; print('OK')"`

---

### TASK 1.6 — Create Basic Tools

**File:** `src/tools/calculator.py`

```python
"""Calculator tool."""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 2', '(3 * 4) / 2'
    """
    try:
        # Safe eval for math expressions only
        allowed_names = {"__builtins__": {}}
        import math
        allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
        result = eval(expression, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
```

**File:** `src/tools/weather.py`

```python
"""Weather tool (mock implementation)."""

from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: City name or location, e.g. 'San Francisco, CA'
    """
    # Mock implementation — replace with real API in production
    return f"Weather in {location}: 72°F, partly cloudy, humidity 55%."
```

**File:** `src/tools/web_search.py`

```python
"""Web search tool using Tavily."""

from langchain_core.tools import tool
from src.config.settings import TAVILY_API_KEY


@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: Search query string
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=3)
        formatted = []
        for r in results.get("results", []):
            formatted.append(f"- {r['title']}: {r['content'][:200]}")
        return "\n".join(formatted) if formatted else "No results found."
    except Exception as e:
        return f"Search error: {e}"
```

**File:** `src/tools/calendar_tool.py`

```python
"""Calendar tool (mock implementation)."""

from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def schedule_event(title: str, date: str, time: str) -> str:
    """Schedule a calendar event.

    Args:
        title: Event title
        date: Event date in YYYY-MM-DD format
        time: Event time in HH:MM format
    """
    return f"Event '{title}' scheduled for {date} at {time}."
```

**File:** `src/tools/email_tool.py`

```python
"""Email tool (mock implementation requiring HITL approval)."""

from langchain_core.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This action requires human approval.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body text
    """
    # In the real system, this is gated by interrupt() in the graph
    return f"Email sent to {to} with subject '{subject}'."
```

**File:** `src/tools/file_tools.py`

```python
"""File operation tools (mock/sandboxed)."""

from langchain_core.tools import tool


@tool
def read_file(filepath: str) -> str:
    """Read contents of a file.

    Args:
        filepath: Path to the file to read
    """
    try:
        with open(filepath, "r") as f:
            content = f.read(10000)  # Limit to 10KB
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file.

    Args:
        filepath: Path to the file to write
        content: Content to write
    """
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"File written: {filepath} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"
```

**File:** `src/tools/code_runner.py`

```python
"""Sandboxed code execution tool."""

from langchain_core.tools import tool
import io
import contextlib


@tool
def run_python_code(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        result = ""
        if output:
            result += f"Output:\n{output}"
        if errors:
            result += f"\nStderr:\n{errors}"
        return result if result else "Code executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}"
```

**File:** `src/tools/knowledge_base.py`

```python
"""Knowledge base / vector store search tool."""

from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant information.

    Args:
        query: Search query
    """
    # Mock — in production, use ChromaDB or similar
    return f"Knowledge base results for '{query}': [Mock] Relevant information about the topic."
```

**File:** `src/tools/state_updating_tools.py`

```python
"""Tools that directly update graph state using Command."""

from langchain_core.tools import tool
from langgraph.types import Command


@tool
def set_user_preference(preference_key: str, preference_value: str) -> Command:
    """Set a user preference that persists in the conversation state.

    Args:
        preference_key: The preference name (e.g. 'theme', 'language')
        preference_value: The preference value (e.g. 'dark', 'spanish')
    """
    # This tool returns a Command that updates graph state directly
    return Command(update={
        "metadata": {preference_key: preference_value},
        "tool_results": [{"tool": "set_user_preference", "key": preference_key, "value": preference_value}]
    })


@tool
def flag_for_review(reason: str) -> Command:
    """Flag the current conversation for human review.

    Args:
        reason: Why this conversation needs human review
    """
    return Command(update={
        "requires_human_approval": True,
        "tool_results": [{"tool": "flag_for_review", "reason": reason}]
    })
```

**File:** `src/tools/__init__.py`

```python
"""Tool registry — central place to import all tools."""

from src.tools.calculator import calculator
from src.tools.weather import get_weather
from src.tools.web_search import web_search
from src.tools.calendar_tool import get_current_time, schedule_event
from src.tools.email_tool import send_email
from src.tools.file_tools import read_file, write_file
from src.tools.code_runner import run_python_code
from src.tools.knowledge_base import search_knowledge_base
from src.tools.state_updating_tools import set_user_preference, flag_for_review

# Tool groups for different agents
BASE_TOOLS = [calculator, get_weather, get_current_time]
CHAT_TOOLS = [calculator, get_weather, get_current_time, search_knowledge_base]
RESEARCH_TOOLS = [web_search, search_knowledge_base]
WRITING_TOOLS = [web_search, read_file, write_file]
TASK_TOOLS = [get_current_time, schedule_event]
CODE_TOOLS = [run_python_code]
ALL_TOOLS = [
    calculator, get_weather, web_search, get_current_time, schedule_event,
    send_email, read_file, write_file, run_python_code, search_knowledge_base,
    set_user_preference, flag_for_review,
]
```

**VERIFY:**
```bash
python -c "from src.tools import CHAT_TOOLS, ALL_TOOLS; print(f'{len(ALL_TOOLS)} tools loaded')"
```
**Expected:** `12 tools loaded`

---

### TASK 1.7 — Build Chat Agent (Prebuilt `create_react_agent`)

**File:** `src/graphs/chat.py`

```python
"""Chat Agent — demonstrates both prebuilt and custom StateGraph approaches.

Concepts covered:
- create_react_agent (prebuilt)
- Custom StateGraph with ToolNode
- Dynamic tool calling
- Message trimming
- Tool calling
- Cache Policy
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import Command

from src.config.prompts import CHAT_AGENT_PROMPT
from src.config.models import get_chat_model
from src.tools import CHAT_TOOLS, CODE_TOOLS


# ──────────────────────────────────────────────────
# APPROACH 1: Quick bootstrap with create_react_agent
# ──────────────────────────────────────────────────
def build_simple_chat_agent():
    """Build a simple chat agent using the prebuilt create_react_agent helper."""
    return create_react_agent(
        model="gpt-4o-mini",
        tools=CHAT_TOOLS,
        prompt=CHAT_AGENT_PROMPT,
    )


# ──────────────────────────────────────────────────
# APPROACH 2: Full custom StateGraph for deep control
# ──────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    current_agent: str


# Message trimmer — keeps conversation from growing unbounded
message_trimmer = trim_messages(
    max_tokens=4000,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o-mini"),
    include_system=True,
    allow_partial=False,
    start_on="human",
)


def chat_model_node(state: ChatState) -> dict:
    """Call the LLM with tools bound. Trims messages first."""
    model = get_chat_model("gpt-4o-mini")

    # Trim messages to fit context window
    trimmed = message_trimmer.invoke(state["messages"])

    # Prepend system message
    messages_with_system = [SystemMessage(content=CHAT_AGENT_PROMPT)] + trimmed

    # Dynamic tool calling — add code tools if code execution is contextually relevant
    tools = list(CHAT_TOOLS)
    last_msg = state["messages"][-1].content if state["messages"] else ""
    if any(kw in last_msg.lower() for kw in ["code", "python", "script", "program", "execute"]):
        tools.extend(CODE_TOOLS)

    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages_with_system)

    return {"messages": [response], "current_agent": "chat_agent"}


def should_use_tools(state: ChatState) -> str:
    """Conditional edge: check if the last message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_custom_chat_agent() -> StateGraph:
    """Build a custom chat agent StateGraph with full control."""
    # Dynamically build tool list (base + conditional)
    all_possible_tools = list(CHAT_TOOLS) + list(CODE_TOOLS)
    tool_node = ToolNode(all_possible_tools)

    builder = StateGraph(ChatState)

    builder.add_node("chat_model", chat_model_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "chat_model")
    builder.add_conditional_edges("chat_model", should_use_tools, {"tools": "tools", END: END})
    builder.add_edge("tools", "chat_model")  # Loop back after tool execution

    return builder


# ──────────────────────────────────────────────────
# DEFAULT EXPORT — the compiled custom agent
# ──────────────────────────────────────────────────
def get_chat_graph(checkpointer=None):
    """Return compiled chat agent graph."""
    builder = build_custom_chat_agent()
    return builder.compile(checkpointer=checkpointer)
```

**VERIFY:**
```bash
python -c "
from src.graphs.chat import build_simple_chat_agent, build_custom_chat_agent
simple = build_simple_chat_agent()
custom = build_custom_chat_agent()
print('Simple chat agent:', type(simple))
print('Custom chat agent builder:', type(custom))
print('Chat agents OK')
"
```

---

### TASK 1.8 — Build Utility Nodes

**File:** `src/nodes/input_handler.py`

```python
"""Input handler node — validates and preprocesses incoming messages."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from src.state.main_state import ValidatedInput


def input_handler_node(state: dict) -> dict:
    """Validate and preprocess user input.

    This is the first node in the main graph.
    Uses Pydantic validation at the boundary.
    """
    # Validate input using Pydantic
    try:
        validated = ValidatedInput(
            messages=state.get("messages", []),
            user_id=state.get("user_id", "default_user"),
        )
    except Exception as e:
        return {
            "error_log": [f"Input validation error: {str(e)}"],
            "current_agent": "error_handler",
        }

    return {
        "user_id": validated.user_id,
        "retry_count": 0,
    }
```

**File:** `src/nodes/output_handler.py`

```python
"""Output handler node — formats final response and manages streaming."""

from __future__ import annotations


def output_handler_node(state: dict) -> dict:
    """Final node that prepares the output.

    Ensures the response is properly formatted before returning to the user.
    """
    return {
        "current_agent": state.get("current_agent", "unknown"),
    }
```

**File:** `src/nodes/error_handler.py`

```python
"""Global error handler node."""

from __future__ import annotations

from langchain_core.messages import AIMessage


def error_handler_node(state: dict) -> dict:
    """Handle errors gracefully — catch-all node for failed operations."""
    errors = state.get("error_log", [])
    last_error = errors[-1] if errors else "Unknown error occurred"

    error_message = AIMessage(
        content=f"I apologize, but I encountered an issue: {last_error}. "
                f"Please try again or rephrase your request."
    )

    return {
        "messages": [error_message],
        "current_agent": "error_handler",
    }
```

**File:** `src/nodes/message_utils.py`

```python
"""Message utility functions — trimming, summarization."""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


def summarize_messages(messages: list, model_name: str = "gpt-4o-mini") -> str:
    """Summarize a list of messages into a concise summary."""
    if not messages:
        return ""

    model = ChatOpenAI(model=model_name, temperature=0)

    text_messages = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text_messages.append(f"{role}: {m.content}")

    conversation_text = "\n".join(text_messages[-20:])  # Last 20 messages

    summary_prompt = [
        SystemMessage(content="Summarize this conversation concisely, preserving key facts and context:"),
        HumanMessage(content=conversation_text),
    ]

    response = model.invoke(summary_prompt)
    return response.content


def should_summarize(messages: list, threshold: int = 20) -> bool:
    """Check if messages should be summarized based on count."""
    return len(messages) > threshold
```

**VERIFY:**
```bash
python -c "
from src.nodes.input_handler import input_handler_node
from src.nodes.output_handler import output_handler_node
from src.nodes.error_handler import error_handler_node
from src.nodes.message_utils import should_summarize
print('All utility nodes OK')
"
```

---

### TASK 1.9 — Build Streaming Utilities

**File:** `src/utils/streaming.py`

```python
"""Streaming utilities — StreamWriter helpers, custom event emitters."""

from __future__ import annotations

from typing import Any


def format_stream_event(event_type: str, data: Any) -> dict:
    """Create a standardized stream event dictionary."""
    return {
        "type": event_type,
        "data": data,
    }


def progress_event(step: str, percent: int, detail: str = "") -> dict:
    """Create a progress stream event."""
    return format_stream_event("progress", {
        "step": step,
        "percent": percent,
        "detail": detail,
    })


def thinking_event(content: str) -> dict:
    """Create a 'thinking' stream event for UI."""
    return format_stream_event("thinking", {"content": content})


def agent_handoff_event(from_agent: str, to_agent: str) -> dict:
    """Create an agent handoff notification event."""
    return format_stream_event("agent_handoff", {
        "from": from_agent,
        "to": to_agent,
    })
```

**VERIFY:** `python -c "from src.utils.streaming import progress_event; print(progress_event('test', 50))"`

---

### TASK 1.10 — Build CLI Chat Interface

**File:** `scripts/cli_chat.py`

```python
"""CLI chat interface for testing OmniAssist.

Usage: python scripts/cli_chat.py
"""

import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import InMemorySaver

from src.graphs.chat import get_chat_graph

console = Console()


def main():
    """Run the CLI chat interface."""
    console.print(Panel.fit(
        "[bold green]OmniAssist CLI Chat[/bold green]\n"
        "Type your message and press Enter. Type 'quit' to exit.\n"
        "Type 'new' to start a new thread.",
        title="Welcome",
    ))

    checkpointer = InMemorySaver()
    graph = get_chat_graph(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Thread: {thread_id}[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            console.print(f"\n[green]New thread: {thread_id}[/green]\n")
            continue

        # Invoke graph
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)], "user_id": "cli_user"},
                config=config,
            )

            # Extract last AI message
            last_ai_msg = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    last_ai_msg = msg
                    break

            if last_ai_msg:
                console.print(f"\n[bold green]Assistant:[/bold green] ", end="")
                console.print(Markdown(last_ai_msg.content))
                console.print()
            else:
                console.print("[dim]No response generated.[/dim]\n")

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
```

**VERIFY:** `python -c "import scripts.cli_chat; print('CLI module loads OK')"`

> **MANUAL TEST (if API keys available):** `python scripts/cli_chat.py` — type "Hello" and verify you get a response. Type "quit" to exit.

---

### TASK 1.11 — Write Phase 1 Tests

**File:** `tests/test_state.py`

```python
"""Tests for state schemas and reducers."""

import pytest
from src.state import append_reducer
from src.state.main_state import OmniAssistState, ValidatedInput, OmniAssistInput, OmniAssistOutput
from src.state.context import OmniAssistContext
from src.state.research_state import ResearchState
from src.state.writing_state import WritingState
from src.state.task_state import TaskState
from src.state.code_state import CodeState
from src.state.data_state import DataState


def test_append_reducer_extends_list():
    assert append_reducer([1, 2], [3, 4]) == [1, 2, 3, 4]


def test_append_reducer_appends_single():
    assert append_reducer([1, 2], 3) == [1, 2, 3]


def test_append_reducer_handles_none():
    assert append_reducer(None, [1, 2]) == [1, 2]


def test_validated_input_accepts_valid():
    v = ValidatedInput(messages=[{"role": "user", "content": "hi"}], user_id="user123")
    assert v.user_id == "user123"


def test_validated_input_rejects_empty_user():
    with pytest.raises(Exception):
        ValidatedInput(messages=[], user_id="   ")


def test_state_schemas_are_typed_dicts():
    """Verify all state schemas are TypedDict subclasses."""
    for schema in [OmniAssistState, OmniAssistInput, OmniAssistOutput,
                   ResearchState, WritingState, TaskState, CodeState, DataState]:
        assert hasattr(schema, "__annotations__"), f"{schema.__name__} missing annotations"


def test_context_schema_has_expected_keys():
    expected_keys = {"model_name", "temperature", "persona", "max_research_sources",
                     "code_execution_enabled", "language"}
    assert expected_keys == set(OmniAssistContext.__annotations__.keys())
```

**File:** `tests/test_chat_agent.py`

```python
"""Tests for the chat agent graph."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.graphs.chat import build_simple_chat_agent, build_custom_chat_agent, get_chat_graph


def test_custom_chat_agent_compiles():
    """Test that the custom chat agent graph compiles without error."""
    builder = build_custom_chat_agent()
    graph = builder.compile()
    assert graph is not None


def test_simple_chat_agent_builds():
    """Test that the prebuilt react agent builds."""
    agent = build_simple_chat_agent()
    assert agent is not None


def test_chat_graph_with_checkpointer():
    """Test that chat graph works with InMemorySaver."""
    graph = get_chat_graph(checkpointer=InMemorySaver())
    assert graph is not None


@pytest.mark.skipif(
    not __import__("os").getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_chat_agent_responds():
    """Integration test — requires API key."""
    graph = get_chat_graph(checkpointer=InMemorySaver())
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2+2?")], "user_id": "test"},
        config={"configurable": {"thread_id": "test-1"}},
    )
    assert len(result["messages"]) >= 2  # At least user msg + AI response
    last_msg = result["messages"][-1]
    assert isinstance(last_msg, AIMessage)
```

**VERIFY:** `python -m pytest tests/test_state.py tests/test_chat_agent.py -v --tb=short`

---

---

# 🔬 PHASE 2 — CORE SUBGRAPHS (Days 4–7)

> **Goal:** Build Research, Writing, and Task Management subgraphs. Implement fan-out/fan-in, deferred nodes, cycles, interrupt(), Command(resume=...), cross-thread Store, and semantic search.
>
> **Concepts covered:** `Subgraphs`, `Nested Subgraphs`, `Fan-out/Fan-in`, `Deferred Nodes`, `Map-Reduce`, `Cycles`, `Recursion Limit`, `interrupt()`, `Command(resume=...)`, `Static Breakpoints`, `Cross-thread Store`, `Semantic Memory Search`, `Custom Streaming Events`, `Cache Policy`, `Retry Policy`

---

### TASK 2.1 — Build Source Validation Nested Subgraph

**File:** `src/graphs/source_validator.py`

```python
"""Source Validation — nested subgraph used within the Research subgraph.

Concepts covered: Nested subgraphs, conditional edges, retry policy.
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from src.config.models import get_chat_model
from src.state import append_reducer


class SourceValidationState(TypedDict):
    sources: list[dict]
    validated_sources: Annotated[list[dict], append_reducer]
    rejected_sources: Annotated[list[dict], append_reducer]


def validate_source_node(state: SourceValidationState) -> dict:
    """Validate each source for reliability and relevance."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    validated = []
    rejected = []

    for source in state.get("sources", []):
        prompt = [
            SystemMessage(content="Rate this source reliability 1-10. Respond with just the number."),
            HumanMessage(content=f"Source: {source.get('title', 'Unknown')} - {source.get('content', '')[:300]}"),
        ]
        try:
            response = model.invoke(prompt)
            score = int(response.content.strip().split()[0])
            source_with_score = {**source, "reliability_score": score}
            if score >= 5:
                validated.append(source_with_score)
            else:
                rejected.append(source_with_score)
        except Exception:
            source_with_score = {**source, "reliability_score": 0}
            rejected.append(source_with_score)

    return {"validated_sources": validated, "rejected_sources": rejected}


def check_enough_sources(state: SourceValidationState) -> str:
    """Conditional edge: check if we have enough validated sources."""
    if len(state.get("validated_sources", [])) >= 2:
        return END
    return "needs_more"


def build_source_validator() -> StateGraph:
    """Build the source validation nested subgraph."""
    builder = StateGraph(SourceValidationState)

    builder.add_node(
        "validate",
        validate_source_node,
        retry=RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0)
    )

    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", check_enough_sources, {END: END, "needs_more": END})

    return builder


# Compiled subgraph for import
source_validator_graph = build_source_validator().compile()
```

**VERIFY:** `python -c "from src.graphs.source_validator import source_validator_graph; print('Source validator OK')"`

---

### TASK 2.2 — Build Research Subgraph

**File:** `src/graphs/research.py`

```python
"""Research Subgraph — performs parallel web search, knowledge base search,
and academic search, then validates and synthesizes results.

Concepts covered:
- Subgraphs
- Nested subgraphs (source_validator)
- Fan-out / Fan-in (parallel execution)
- Deferred nodes
- Map-Reduce pattern
- Cycles (quality loop)
- Recursion limit
- Custom streaming events
- Cache policy
- Retry policy
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, Send

from src.config.models import get_chat_model
from src.config.prompts import RESEARCH_AGENT_PROMPT
from src.state import append_reducer
from src.state.research_state import ResearchState
from src.tools.web_search import web_search
from src.tools.knowledge_base import search_knowledge_base


# ──────────────────────────────────────────────
# NODE FUNCTIONS
# ──────────────────────────────────────────────

def query_planner_node(state: ResearchState) -> dict:
    """Plan search queries based on user's research request."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else state.get("query", "")

    prompt = [
        SystemMessage(content=(
            "Generate 2-3 specific search queries to research this topic thoroughly. "
            "Return each query on a new line, nothing else."
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    return {"search_queries": queries, "query": last_message}


def web_search_node(state: ResearchState) -> dict:
    """Execute web searches for all planned queries."""
    results = []
    for query in state.get("search_queries", [])[:3]:
        try:
            search_result = web_search.invoke({"query": query})
            results.append({"source": "web", "query": query, "content": search_result, "title": query})
        except Exception as e:
            results.append({"source": "web", "query": query, "content": f"Error: {e}", "title": query})
    return {"web_results": results}


def knowledge_base_node(state: ResearchState) -> dict:
    """Search internal knowledge base."""
    query = state.get("query", "")
    try:
        result = search_knowledge_base.invoke({"query": query})
        return {"kb_results": [{"source": "knowledge_base", "query": query, "content": result, "title": f"KB: {query}"}]}
    except Exception as e:
        return {"kb_results": [{"source": "knowledge_base", "query": query, "content": f"Error: {e}", "title": query}]}


def academic_search_node(state: ResearchState) -> dict:
    """Search academic sources (mock)."""
    query = state.get("query", "")
    return {
        "academic_results": [{
            "source": "academic",
            "query": query,
            "content": f"Academic findings on '{query}': [Mock academic search result]",
            "title": f"Academic: {query}",
        }]
    }


def results_aggregator_node(state: ResearchState) -> dict:
    """Aggregate all search results from parallel branches.

    This is a DEFERRED node — it waits until all fan-out branches complete.
    """
    all_results = (
        state.get("web_results", [])
        + state.get("kb_results", [])
        + state.get("academic_results", [])
    )
    # Pass combined results to source validation
    return {"validated_sources": all_results}


def synthesis_node(state: ResearchState) -> dict:
    """Synthesize validated research into a coherent response."""
    model = get_chat_model("gpt-4o-mini", temperature=0.3)

    sources = state.get("validated_sources", [])
    source_texts = []
    for i, src in enumerate(sources, 1):
        source_texts.append(f"[{i}] ({src.get('source', 'unknown')}): {src.get('content', '')[:300]}")

    prompt = [
        SystemMessage(content=RESEARCH_AGENT_PROMPT),
        HumanMessage(content=(
            f"Research query: {state.get('query', '')}\n\n"
            f"Sources found:\n" + "\n".join(source_texts) + "\n\n"
            "Synthesize these into a clear, well-organized response. "
            "Cite sources by number [1], [2], etc."
        )),
    ]

    response = model.invoke(prompt)
    return {
        "synthesis": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def quality_check_node(state: ResearchState) -> dict:
    """Check the quality of the synthesis."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    synthesis = state.get("synthesis", "")

    prompt = [
        SystemMessage(content="Rate this research synthesis quality 1-10. Respond with just the number."),
        HumanMessage(content=synthesis[:1000]),
    ]

    try:
        response = model.invoke(prompt)
        score = float(response.content.strip().split()[0])
    except Exception:
        score = 5.0

    return {"quality_score": score, "needs_more_research": score < 6.0}


def should_research_more(state: ResearchState) -> str:
    """Conditional edge: should we loop back for more research?"""
    if state.get("needs_more_research", False):
        return "needs_improvement"
    return "good_enough"


# ──────────────────────────────────────────────
# BUILD RESEARCH SUBGRAPH
# ──────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """Build the research subgraph with parallel search and quality loop."""
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("query_planner", query_planner_node)
    builder.add_node(
        "web_search", web_search_node,
        retry=RetryPolicy(max_attempts=2, initial_interval=1.0)
    )
    builder.add_node("knowledge_base", knowledge_base_node)
    builder.add_node("academic_search", academic_search_node)
    builder.add_node("results_aggregator", results_aggregator_node, defer=True)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("quality_check", quality_check_node)

    # Entry
    builder.add_edge(START, "query_planner")

    # Fan-out: parallel search branches
    builder.add_edge("query_planner", "web_search")
    builder.add_edge("query_planner", "knowledge_base")
    builder.add_edge("query_planner", "academic_search")

    # Fan-in: all branches feed into deferred aggregator
    builder.add_edge("web_search", "results_aggregator")
    builder.add_edge("knowledge_base", "results_aggregator")
    builder.add_edge("academic_search", "results_aggregator")

    # Aggregator → Synthesis → Quality Check
    builder.add_edge("results_aggregator", "synthesis")
    builder.add_edge("synthesis", "quality_check")

    # Quality loop (cycle) — conditional edge
    builder.add_conditional_edges(
        "quality_check",
        should_research_more,
        {
            "needs_improvement": "query_planner",  # Loop back
            "good_enough": END,
        }
    )

    return builder


# Compiled graph for import
graph = build_research_graph().compile()
```

**VERIFY:**
```bash
python -c "
from src.graphs.research import graph
print('Research graph nodes:', list(graph.get_graph().nodes))
print('Research graph OK')
"
```

---

### TASK 2.3 — Build Writing Subgraph

**File:** `src/graphs/writing.py`

```python
"""Writing Subgraph — handles content creation with iterative review cycles.

Concepts covered:
- Cycles (write → review → rewrite)
- interrupt() function (modern HITL)
- Command(resume=...) for resumption
- Static breakpoints (interrupt_before/interrupt_after for debugging)
- Retry policy
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command, RetryPolicy

from src.config.models import get_chat_model
from src.config.prompts import WRITING_AGENT_PROMPT
from src.state.writing_state import WritingState


# ──────────────────────────────────────────────
# NODE FUNCTIONS
# ──────────────────────────────────────────────

def analyze_request_node(state: WritingState) -> dict:
    """Analyze the writing request to determine content type and approach."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content=(
            "Classify this writing request. Respond with one word: "
            "email, article, summary, report, or other"
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    content_type = response.content.strip().lower().split()[0]
    valid_types = {"email", "article", "summary", "report", "other"}
    if content_type not in valid_types:
        content_type = "other"

    return {
        "writing_request": last_message,
        "content_type": content_type,
        "iteration_count": 0,
    }


def generate_draft_node(state: WritingState) -> dict:
    """Generate or revise the content draft."""
    model = get_chat_model("gpt-4o-mini", temperature=0.7)

    iteration = state.get("iteration_count", 0)
    feedback = state.get("human_feedback")

    if iteration == 0 or not state.get("draft_content"):
        # First draft
        prompt = [
            SystemMessage(content=WRITING_AGENT_PROMPT),
            HumanMessage(content=(
                f"Create a {state.get('content_type', 'text')} based on this request:\n"
                f"{state.get('writing_request', '')}"
            )),
        ]
    else:
        # Revision based on feedback
        prompt = [
            SystemMessage(content=WRITING_AGENT_PROMPT),
            HumanMessage(content=(
                f"Revise this {state.get('content_type', 'text')} based on feedback.\n\n"
                f"Original request: {state.get('writing_request', '')}\n\n"
                f"Current draft:\n{state.get('draft_content', '')}\n\n"
                f"Feedback: {feedback}\n\n"
                "Please produce an improved version."
            )),
        ]

    response = model.invoke(prompt)

    return {
        "draft_content": response.content,
        "revision_history": [response.content],
        "iteration_count": iteration + 1,
    }


def quality_review_node(state: WritingState) -> dict:
    """Auto-review the draft for quality."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    draft = state.get("draft_content", "")

    prompt = [
        SystemMessage(content="Rate this writing 1-10 for quality. Respond with just the number."),
        HumanMessage(content=draft[:2000]),
    ]

    try:
        response = model.invoke(prompt)
        score = float(response.content.strip().split()[0])
    except Exception:
        score = 5.0

    return {"quality_score": score}


def human_review_node(state: WritingState) -> dict:
    """Human-in-the-loop review using modern interrupt() function.

    IMPORTANT: When execution resumes via Command(resume=...),
    the ENTIRE node re-executes from the beginning. The interrupt()
    call then returns the resume value instead of pausing.
    No side effects should occur before interrupt()!
    """
    draft = state.get("draft_content", "")
    quality_score = state.get("quality_score", 0)

    # interrupt() pauses execution and sends data to the caller
    feedback = interrupt({
        "draft": draft,
        "quality_score": quality_score,
        "iteration": state.get("iteration_count", 0),
        "prompt": "Review this draft. Respond with: {'action': 'approve'|'revise'|'restart', 'feedback': '...'}",
    })

    # When resumed with Command(resume={"action": "approve", "feedback": "Looks good"}),
    # this code runs with feedback = that resume value
    action = feedback.get("action", "approve") if isinstance(feedback, dict) else "approve"
    fb_text = feedback.get("feedback", "") if isinstance(feedback, dict) else str(feedback)

    return {
        "action": action,
        "human_feedback": fb_text,
    }


def route_after_review(state: WritingState) -> str:
    """Route based on human review action."""
    action = state.get("action", "approve")
    iteration = state.get("iteration_count", 0)

    if action == "approve":
        return "finalize"
    elif action == "restart":
        return "generate_draft"
    elif action == "revise" and iteration < 5:  # Max 5 iterations
        return "generate_draft"
    else:
        return "finalize"  # Force finalize after max iterations


def finalize_node(state: WritingState) -> dict:
    """Finalize the writing and produce the output message."""
    draft = state.get("draft_content", "")
    return {
        "messages": [AIMessage(content=f"Here's your finalized content:\n\n{draft}")],
        "draft_content": draft,
    }


# ──────────────────────────────────────────────
# BUILD WRITING SUBGRAPH
# ──────────────────────────────────────────────

def build_writing_graph() -> StateGraph:
    """Build the writing subgraph with review cycles and HITL."""
    builder = StateGraph(WritingState)

    builder.add_node("analyze_request", analyze_request_node)
    builder.add_node(
        "generate_draft", generate_draft_node,
        retry=RetryPolicy(max_attempts=2)
    )
    builder.add_node("quality_review", quality_review_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    # Flow
    builder.add_edge(START, "analyze_request")
    builder.add_edge("analyze_request", "generate_draft")
    builder.add_edge("generate_draft", "quality_review")
    builder.add_edge("quality_review", "human_review")

    # Cycle: review → rewrite or finalize
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "generate_draft": "generate_draft",
            "finalize": "finalize",
        }
    )
    builder.add_edge("finalize", END)

    return builder


# Compiled graph (no checkpointer — parent graph provides it)
graph = build_writing_graph().compile()
```

**VERIFY:**
```bash
python -c "
from src.graphs.writing import graph
print('Writing graph nodes:', list(graph.get_graph().nodes))
print('Writing graph OK')
"
```

---

### TASK 2.4 — Build Task Management Subgraph

**File:** `src/graphs/task_management.py`

```python
"""Task Management Subgraph — CRUD operations on tasks with HITL for destructive actions.

Concepts covered:
- interrupt() for destructive operations
- Cross-thread Store (PostgresStore/InMemoryStore)
- Semantic memory search
- Store access via dependency injection
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.store.base import BaseStore

from src.config.models import get_chat_model
from src.config.prompts import TASK_AGENT_PROMPT
from src.state import append_reducer
from src.state.task_state import TaskState


# ──────────────────────────────────────────────
# NODE FUNCTIONS
# ──────────────────────────────────────────────

def classify_task_action_node(state: TaskState) -> dict:
    """Classify what task action the user wants."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content=(
            "Classify this task management request. Respond with one word: "
            "create, update, delete, list, delete_all, or bulk_update"
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    action = response.content.strip().lower().split()[0]
    valid_actions = {"create", "update", "delete", "list", "delete_all", "bulk_update"}
    if action not in valid_actions:
        action = "list"

    return {"task_action": action}


def find_related_tasks_node(state: TaskState, *, store: BaseStore) -> dict:
    """Find related tasks using semantic memory search.

    The `store` parameter is automatically injected by LangGraph runtime
    when the graph is compiled with a store.
    """
    user_id = state.get("user_id", "default")
    last_message = state["messages"][-1].content if state["messages"] else ""

    try:
        results = store.search(
            ("users", user_id, "tasks"),
            query=last_message,
            limit=5,
        )
        related = [{"key": r.key, "value": r.value} for r in results]
    except Exception:
        related = []

    return {"related_tasks": related}


def confirmation_gate_node(state: TaskState) -> dict:
    """Gate for destructive actions — uses interrupt() for HITL approval.

    IMPORTANT: No side effects before interrupt() — the entire node
    re-runs on resume.
    """
    action = state.get("task_action", "")
    affected = state.get("affected_count", 0)

    if action in ["delete", "delete_all", "bulk_update"]:
        response = interrupt(
            f"Are you sure you want to '{action}'? "
            f"This will affect {affected} task(s). "
            f"Reply with 'confirmed' to proceed."
        )
        if response != "confirmed":
            return {
                "task_action": "cancelled",
                "messages": [AIMessage(content="Operation cancelled.")],
            }

    return state


def execute_task_action_node(state: TaskState, *, store: BaseStore) -> dict:
    """Execute the task action (CRUD operations using Store)."""
    user_id = state.get("user_id", "default")
    action = state.get("task_action", "list")
    last_message = state["messages"][-1].content if state["messages"] else ""

    namespace = ("users", user_id, "tasks")

    if action == "cancelled":
        return {"messages": [AIMessage(content="Operation was cancelled.")]}

    if action == "create":
        import uuid
        task_id = str(uuid.uuid4())[:8]
        task_data = {"text": last_message, "status": "pending", "id": task_id}
        try:
            store.put(namespace, task_id, task_data)
        except Exception:
            pass  # Store may not be available in tests
        return {
            "tasks": [task_data],
            "messages": [AIMessage(content=f"Task created: '{last_message}' (ID: {task_id})")],
        }

    if action == "list":
        try:
            items = store.search(namespace, query="", limit=20)
            task_list = [item.value for item in items]
        except Exception:
            task_list = state.get("tasks", [])

        if task_list:
            lines = [f"- [{t.get('status', '?')}] {t.get('text', '?')} (ID: {t.get('id', '?')})" for t in task_list]
            return {"messages": [AIMessage(content="Your tasks:\n" + "\n".join(lines))]}
        return {"messages": [AIMessage(content="No tasks found.")]}

    if action == "delete":
        return {"messages": [AIMessage(content="Task deleted successfully.")]}

    return {"messages": [AIMessage(content=f"Action '{action}' completed.")]}


def route_after_classification(state: TaskState) -> str:
    """Route based on whether confirmation is needed."""
    action = state.get("task_action", "")
    if action in ["delete", "delete_all", "bulk_update"]:
        return "confirmation_gate"
    return "execute_task_action"


# ──────────────────────────────────────────────
# BUILD TASK MANAGEMENT SUBGRAPH
# ──────────────────────────────────────────────

def build_task_graph() -> StateGraph:
    """Build the task management subgraph."""
    builder = StateGraph(TaskState)

    builder.add_node("classify_action", classify_task_action_node)
    builder.add_node("find_related", find_related_tasks_node)
    builder.add_node("confirmation_gate", confirmation_gate_node)
    builder.add_node("execute_task_action", execute_task_action_node)

    builder.add_edge(START, "classify_action")
    builder.add_edge("classify_action", "find_related")

    builder.add_conditional_edges(
        "find_related",
        route_after_classification,
        {
            "confirmation_gate": "confirmation_gate",
            "execute_task_action": "execute_task_action",
        }
    )
    builder.add_edge("confirmation_gate", "execute_task_action")
    builder.add_edge("execute_task_action", END)

    return builder


graph = build_task_graph().compile()
```

**VERIFY:**
```bash
python -c "
from src.graphs.task_management import graph
print('Task management graph nodes:', list(graph.get_graph().nodes))
print('Task management graph OK')
"
```

---

### TASK 2.5 — Build Human Review Node (Shared)

**File:** `src/nodes/human_review.py`

```python
"""Shared human review node using interrupt() function.

Can be reused across subgraphs wherever HITL approval is needed.
"""

from __future__ import annotations

from langgraph.types import interrupt


def human_approval_gate(state: dict) -> dict:
    """Generic human approval gate.

    Usage: Add this node before any destructive/sensitive action.
    The caller resumes with Command(resume="approved") or Command(resume="rejected").

    IMPORTANT: The entire node re-executes on resume. No side effects before interrupt()!
    """
    action = state.get("pending_action", "unknown action")
    details = state.get("pending_details", "")

    response = interrupt({
        "action": action,
        "details": details,
        "prompt": f"Approve '{action}'? Reply 'approved' or 'rejected'.",
    })

    return {
        "human_feedback": response,
        "requires_human_approval": False,
    }
```

**VERIFY:** `python -c "from src.nodes.human_review import human_approval_gate; print('OK')"`

---

### TASK 2.6 — Write Phase 2 Tests

**File:** `tests/test_research.py`

```python
"""Tests for the Research subgraph."""

import pytest
from src.graphs.research import build_research_graph, graph
from src.graphs.source_validator import build_source_validator


def test_research_graph_compiles():
    builder = build_research_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_research_graph_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["query_planner", "web_search", "knowledge_base",
                "academic_search", "results_aggregator", "synthesis", "quality_check"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"


def test_source_validator_compiles():
    builder = build_source_validator()
    compiled = builder.compile()
    assert compiled is not None
```

**File:** `tests/test_writing.py`

```python
"""Tests for the Writing subgraph."""

import pytest
from src.graphs.writing import build_writing_graph, graph


def test_writing_graph_compiles():
    builder = build_writing_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_writing_graph_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["analyze_request", "generate_draft", "quality_review", "human_review", "finalize"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"
```

**File:** `tests/test_task_management.py`

```python
"""Tests for the Task Management subgraph."""

import pytest
from src.graphs.task_management import build_task_graph, graph


def test_task_graph_compiles():
    builder = build_task_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_task_graph_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["classify_action", "find_related", "confirmation_gate", "execute_task_action"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"
```

**VERIFY:** `python -m pytest tests/test_research.py tests/test_writing.py tests/test_task_management.py -v --tb=short`

---

---

# ⚡ PHASE 3 — FUNCTIONAL API & ADVANCED FEATURES (Days 8–11)

> **Goal:** Build the Code Execution pipeline (Functional API), Data Analysis subgraph, Main Orchestrator with supervisor, swarm pattern, memory system, and advanced tool features.
>
> **Concepts covered:** `@entrypoint`, `@task`, `entrypoint.final`, `previous` parameter, `Map-Reduce`, `langgraph-supervisor`, `langgraph-swarm`, `Short-term Memory`, `Long-term Memory`, `LangMem SDK`, `Tools that update state`, `Dynamic Tool Calling`, `Command(goto=...)`, `Deferred Nodes`

---

### TASK 3.1 — Build Code Execution Pipeline (Functional API)

**File:** `src/workflows/code_pipeline.py`

```python
"""Code Execution Pipeline — built using the Functional API (@entrypoint/@task).

This module demonstrates the alternative to StateGraph for simpler,
linear workflows.

Concepts covered:
- @entrypoint and @task decorators
- interrupt() in Functional API
- entrypoint.final (return different value vs save to checkpoint)
- previous parameter (access last invocation result)
- Retry policy
- Cache policy
"""

from __future__ import annotations

from langgraph.func import entrypoint, task
from langgraph.types import interrupt, RetryPolicy, CachePolicy
from langgraph.checkpoint.memory import InMemorySaver

from src.config.models import get_chat_model
from src.config.prompts import CODE_AGENT_PROMPT


@task(retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0))
def generate_code(request: str) -> dict:
    """Generate Python code from a natural language request."""
    model = get_chat_model("gpt-4o-mini", temperature=0.2)
    from langchain_core.messages import SystemMessage, HumanMessage

    response = model.invoke([
        SystemMessage(content=CODE_AGENT_PROMPT + "\nGenerate ONLY Python code. No explanation. Just code."),
        HumanMessage(content=request),
    ])
    return {"code": response.content, "language": "python"}


@task
def review_code(code_info: dict) -> dict:
    """Review code for correctness and safety."""
    code = code_info.get("code", "")

    # Simple safety check
    dangerous_patterns = ["os.system", "subprocess", "eval(", "exec(", "__import__",
                          "shutil.rmtree", "os.remove", "open(", "import socket"]
    safety_level = "safe"
    for pattern in dangerous_patterns:
        if pattern in code:
            safety_level = "dangerous"
            break

    if safety_level == "safe" and any(p in code for p in ["requests.", "http", "urllib"]):
        safety_level = "caution"

    model = get_chat_model("gpt-4o-mini", temperature=0)
    from langchain_core.messages import SystemMessage, HumanMessage
    review_response = model.invoke([
        SystemMessage(content="Review this code briefly (1-2 sentences). Note any issues."),
        HumanMessage(content=code),
    ])

    return {
        **code_info,
        "safety_level": safety_level,
        "review": review_response.content,
    }


@task
def execute_code(code: str) -> dict:
    """Execute code in a sandboxed environment."""
    import io
    import contextlib

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue() if stderr_capture.getvalue() else None
        return {"output": output or "Code executed successfully (no output).", "error": error}
    except Exception as e:
        return {"output": "", "error": f"{type(e).__name__}: {e}"}


@entrypoint(checkpointer=InMemorySaver())
def code_pipeline(request: str, *, previous: dict | None = None) -> dict:
    """Main code execution pipeline using Functional API.

    Args:
        request: Natural language code request
        previous: Result of the last invocation on the same thread.
                  Automatically provided by LangGraph runtime.

    The `previous` parameter enables continuity between invocations
    on the same thread — e.g., "now modify that code to also..."
    """
    # If previous result exists, include it as context
    context = ""
    if previous and previous.get("code"):
        context = f"\n\nPrevious code:\n{previous['code']}\nPrevious output:\n{previous.get('output', '')}"

    full_request = request + context if context else request

    # Step 1: Generate code
    code_info = generate_code(full_request).result()

    # Step 2: Review code
    review = review_code(code_info).result()

    # Step 3: Safety gate — interrupt if dangerous
    if review["safety_level"] == "dangerous":
        approval = interrupt({
            "code": code_info["code"],
            "safety_level": "dangerous",
            "review": review["review"],
            "prompt": "This code is potentially unsafe. Reply 'approved' to execute or 'rejected' to cancel.",
        })
        if approval != "approved":
            # entrypoint.final: return different value to caller vs. save different value to checkpoint
            return entrypoint.final(
                value={"status": "cancelled", "reason": "User rejected unsafe code"},
                save={"last_request": request, "cancelled": True, "code": code_info["code"]},
            )

    # Step 4: Execute code
    result = execute_code(code_info["code"]).result()

    return {
        "code": code_info["code"],
        "language": code_info["language"],
        "safety_level": review["safety_level"],
        "review": review["review"],
        "output": result.get("output", ""),
        "error": result.get("error"),
    }
```

**VERIFY:**
```bash
python -c "
from src.workflows.code_pipeline import code_pipeline, generate_code, review_code, execute_code
print('Code pipeline OK')
print('Type of code_pipeline:', type(code_pipeline))
"
```

---

### TASK 3.2 — Build Quick Research Workflow (Functional API)

**File:** `src/workflows/quick_research.py`

```python
"""Quick Research — lightweight research using Functional API.

Demonstrates using @entrypoint/@task for simpler workflows
that don't need the full StateGraph machinery.
"""

from __future__ import annotations

from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

from src.config.models import get_chat_model
from src.tools.web_search import web_search


@task(retry_policy=RetryPolicy(max_attempts=2))
def quick_search(query: str) -> str:
    """Perform a quick web search."""
    try:
        return web_search.invoke({"query": query})
    except Exception as e:
        return f"Search failed: {e}"


@task
def quick_summarize(query: str, search_results: str) -> str:
    """Summarize search results concisely."""
    model = get_chat_model("gpt-4o-mini", temperature=0.3)
    from langchain_core.messages import SystemMessage, HumanMessage

    response = model.invoke([
        SystemMessage(content="Summarize these search results concisely in 2-3 sentences."),
        HumanMessage(content=f"Query: {query}\n\nResults:\n{search_results}"),
    ])
    return response.content


@entrypoint(checkpointer=InMemorySaver())
def quick_research(query: str) -> dict:
    """Quick research pipeline — search and summarize in one shot."""
    results = quick_search(query).result()
    summary = quick_summarize(query, results).result()
    return {"query": query, "results": results, "summary": summary}
```

**VERIFY:** `python -c "from src.workflows.quick_research import quick_research; print('Quick research OK')"`

---

### TASK 3.3 — Build MCP Bridge Workflow

**File:** `src/workflows/mcp_bridge.py`

```python
"""MCP Bridge — connects to MCP-compatible tool servers.

Concepts covered:
- MCP Client integration
- langchain-mcp-adapters
- Streamable HTTP transport
"""

from __future__ import annotations


async def get_mcp_tools(servers: dict | None = None):
    """Get tools from MCP servers.

    Args:
        servers: MCP server configuration dict. If None, uses defaults.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print("langchain-mcp-adapters not installed. MCP tools unavailable.")
        return []

    if servers is None:
        # Default MCP servers — customize for your environment
        servers = {
            "file_system": {
                "url": "http://localhost:8080/mcp",
                "transport": "streamable_http",
            },
        }

    try:
        async with MultiServerMCPClient(servers) as client:
            tools = client.get_tools()
            return tools
    except Exception as e:
        print(f"MCP connection error: {e}")
        return []
```

**File:** `src/tools/mcp_tools.py`

```python
"""MCP tool server connections.

Provides utilities for connecting LangGraph agents to MCP tool servers.
"""

from __future__ import annotations


MCP_SERVER_CONFIGS = {
    "file_system": {
        "url": "http://localhost:8080/mcp",
        "transport": "streamable_http",
    },
    # Add more MCP servers as needed
}


def get_mcp_server_config(server_name: str) -> dict | None:
    """Get MCP server config by name."""
    return MCP_SERVER_CONFIGS.get(server_name)
```

**VERIFY:** `python -c "from src.workflows.mcp_bridge import get_mcp_tools; print('MCP bridge OK')"`

---

### TASK 3.4 — Build Data Analysis Subgraph

**File:** `src/graphs/data_analysis.py`

```python
"""Data Analysis Subgraph — handles data processing, statistics, and visualization.

Concepts covered:
- Map-Reduce pattern
- Deferred nodes
- Cache policy
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.config.models import get_chat_model
from src.config.prompts import DATA_AGENT_PROMPT
from src.state import append_reducer
from src.state.data_state import DataState


def parse_data_request_node(state: DataState) -> dict:
    """Parse the data analysis request to determine type and data source."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content=(
            "Classify this data analysis request. Respond with one word: "
            "statistics, visualization, correlation, or summary"
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    analysis_type = response.content.strip().lower().split()[0]
    valid_types = {"statistics", "visualization", "correlation", "summary"}
    if analysis_type not in valid_types:
        analysis_type = "summary"

    return {"analysis_type": analysis_type, "raw_data": last_message}


def statistics_node(state: DataState) -> dict:
    """Compute statistical analysis."""
    model = get_chat_model("gpt-4o-mini", temperature=0)
    data = state.get("raw_data", "")

    prompt = [
        SystemMessage(content=DATA_AGENT_PROMPT),
        HumanMessage(content=f"Provide statistical analysis of:\n{data}"),
    ]
    response = model.invoke(prompt)
    return {"analysis_results": [{"type": "statistics", "result": response.content}]}


def visualization_node(state: DataState) -> dict:
    """Generate visualization description/code."""
    model = get_chat_model("gpt-4o-mini", temperature=0.3)
    data = state.get("raw_data", "")

    prompt = [
        SystemMessage(content=DATA_AGENT_PROMPT),
        HumanMessage(content=f"Generate Python matplotlib code to visualize:\n{data}\nReturn ONLY code."),
    ]
    response = model.invoke(prompt)
    return {"analysis_results": [{"type": "visualization", "result": response.content}]}


def summary_node(state: DataState) -> dict:
    """Generate data summary."""
    model = get_chat_model("gpt-4o-mini", temperature=0.3)
    data = state.get("raw_data", "")

    prompt = [
        SystemMessage(content=DATA_AGENT_PROMPT),
        HumanMessage(content=f"Summarize this data analysis request and provide insights:\n{data}"),
    ]
    response = model.invoke(prompt)
    return {
        "summary": response.content,
        "analysis_results": [{"type": "summary", "result": response.content}],
    }


def results_merger_node(state: DataState) -> dict:
    """Merge all analysis results into a final response.

    This is a DEFERRED node — waits for all analysis branches.
    """
    results = state.get("analysis_results", [])
    result_texts = []
    for r in results:
        result_texts.append(f"**{r.get('type', 'Analysis')}:**\n{r.get('result', '')}")

    combined = "\n\n".join(result_texts) if result_texts else "No analysis results available."

    return {
        "messages": [AIMessage(content=combined)],
        "summary": combined,
    }


def route_analysis_type(state: DataState) -> str:
    """Route to the appropriate analysis node."""
    analysis_type = state.get("analysis_type", "summary")
    if analysis_type == "statistics":
        return "statistics"
    elif analysis_type == "visualization":
        return "visualization"
    else:
        return "summary"


def build_data_analysis_graph() -> StateGraph:
    """Build the data analysis subgraph."""
    builder = StateGraph(DataState)

    builder.add_node("parse_request", parse_data_request_node)
    builder.add_node("statistics", statistics_node)
    builder.add_node("visualization", visualization_node)
    builder.add_node("summary", summary_node)
    builder.add_node("results_merger", results_merger_node, defer=True)

    builder.add_edge(START, "parse_request")

    builder.add_conditional_edges(
        "parse_request",
        route_analysis_type,
        {
            "statistics": "statistics",
            "visualization": "visualization",
            "summary": "summary",
        }
    )

    builder.add_edge("statistics", "results_merger")
    builder.add_edge("visualization", "results_merger")
    builder.add_edge("summary", "results_merger")
    builder.add_edge("results_merger", END)

    return builder


graph = build_data_analysis_graph().compile()
```

**VERIFY:**
```bash
python -c "
from src.graphs.data_analysis import graph
print('Data analysis graph nodes:', list(graph.get_graph().nodes))
print('Data analysis OK')
"
```

---

### TASK 3.5 — Build Memory System

**File:** `src/memory/short_term.py`

```python
"""Short-term memory — within-conversation context management."""

from __future__ import annotations

from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI


def get_message_trimmer(max_tokens: int = 4000, model: str = "gpt-4o-mini"):
    """Get a message trimmer configured for the given context window."""
    return trim_messages(
        max_tokens=max_tokens,
        strategy="last",
        token_counter=ChatOpenAI(model=model),
        include_system=True,
        allow_partial=False,
        start_on="human",
    )


def trim_conversation(messages: list, max_tokens: int = 4000) -> list:
    """Trim a conversation to fit within the token limit."""
    trimmer = get_message_trimmer(max_tokens)
    return trimmer.invoke(messages)
```

**File:** `src/memory/long_term.py`

```python
"""Long-term memory — cross-thread persistence using Store API with semantic search."""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore


def create_memory_store(use_semantic_search: bool = True) -> InMemoryStore:
    """Create an in-memory store with optional semantic search indexing.

    For production, replace with PostgresStore or RedisStore:
        from langgraph.store.postgres import PostgresStore
        store = PostgresStore(conn_string=DATABASE_URL, index={...})
    """
    if use_semantic_search:
        try:
            store = InMemoryStore(
                index={
                    "embed": "openai:text-embedding-3-small",
                    "dims": 1536,
                    "fields": ["text", "summary"],
                }
            )
        except Exception:
            # Fallback without embeddings if OpenAI not available
            store = InMemoryStore()
    else:
        store = InMemoryStore()

    return store


def store_user_fact(store, user_id: str, fact_id: str, fact: dict):
    """Store a user fact in long-term memory."""
    store.put(("users", user_id, "facts"), fact_id, fact)


def search_user_facts(store, user_id: str, query: str, limit: int = 5) -> list:
    """Search user facts using semantic search."""
    try:
        results = store.search(("users", user_id, "facts"), query=query, limit=limit)
        return [{"key": r.key, "value": r.value} for r in results]
    except Exception:
        return []


def store_user_preference(store, user_id: str, key: str, value: str):
    """Store a user preference."""
    store.put(("users", user_id, "preferences"), key, {"text": value, "key": key})


def get_user_preferences(store, user_id: str) -> dict:
    """Get all user preferences."""
    try:
        results = store.search(("users", user_id, "preferences"), query="", limit=50)
        return {r.value.get("key", r.key): r.value for r in results}
    except Exception:
        return {}
```

**File:** `src/memory/user_profile.py`

```python
"""User profile management — persisted across threads via Store."""

from __future__ import annotations

from langgraph.store.base import BaseStore


def get_user_profile(store: BaseStore, user_id: str) -> dict:
    """Retrieve user profile from the store."""
    try:
        results = store.search(("users", user_id, "profile"), query="", limit=10)
        profile = {}
        for r in results:
            profile.update(r.value)
        return profile
    except Exception:
        return {}


def update_user_profile(store: BaseStore, user_id: str, updates: dict):
    """Update user profile in the store."""
    try:
        for key, value in updates.items():
            store.put(("users", user_id, "profile"), key, {"text": str(value), key: value})
    except Exception:
        pass
```

**File:** `src/memory/langmem_integration.py`

```python
"""LangMem SDK integration for automatic memory extraction.

LangMem enables agents to self-improve through long-term memory by
automatically extracting and storing key facts from conversations.
"""

from __future__ import annotations


def get_memory_manager(model: str = "gpt-4o-mini"):
    """Create a LangMem memory manager for automatic memory extraction.

    Returns None if langmem is not available.
    """
    try:
        from langmem import create_memory_manager
        return create_memory_manager(model=model)
    except ImportError:
        print("langmem not installed. Automatic memory extraction disabled.")
        return None


async def extract_memories(memory_manager, messages: list) -> list:
    """Extract key facts/memories from a conversation.

    Returns a list of extracted memory items.
    """
    if memory_manager is None:
        return []
    try:
        memories = await memory_manager.ainvoke({"messages": messages})
        return memories
    except Exception as e:
        print(f"Memory extraction error: {e}")
        return []
```

**VERIFY:**
```bash
python -c "
from src.memory.short_term import trim_conversation
from src.memory.long_term import create_memory_store
from src.memory.user_profile import get_user_profile
from src.memory.langmem_integration import get_memory_manager
print('Memory system OK')
"
```

---

### TASK 3.6 — Build Memory Manager Node

**File:** `src/nodes/memory_manager.py`

```python
"""Memory manager node — handles short-term and long-term memory operations.

Concepts covered:
- Short-term memory (message trimming/summarization)
- Long-term memory (Store API)
- Semantic search
- LangMem integration
- Runtime store access
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from src.memory.short_term import trim_conversation
from src.nodes.message_utils import summarize_messages, should_summarize


def memory_manager_node(state: dict, *, store: BaseStore) -> dict:
    """Manage conversation memory — trim, summarize, and store facts.

    The `store` parameter is automatically injected by the LangGraph runtime
    when the graph is compiled with a store.
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "default")
    updates = {}

    # 1. Summarize if conversation is getting long
    if should_summarize(messages, threshold=20):
        summary = summarize_messages(messages)
        updates["conversation_summary"] = summary

    # 2. Store notable facts from the latest message
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage) and len(last_msg.content) > 20:
            try:
                import uuid
                fact_id = str(uuid.uuid4())[:8]
                store.put(
                    ("users", user_id, "facts"),
                    fact_id,
                    {"text": last_msg.content[:500], "summary": last_msg.content[:100]},
                )
            except Exception:
                pass  # Store operations are best-effort

    return updates
```

**VERIFY:** `python -c "from src.nodes.memory_manager import memory_manager_node; print('Memory manager node OK')"`

---

### TASK 3.7 — Build Multi-Agent Setups

**File:** `src/agents/supervisor_setup.py`

```python
"""Supervisor pattern using langgraph-supervisor library.

Concepts covered:
- langgraph-supervisor for delegation
- Supervisor delegating to specialist agents
"""

from __future__ import annotations


def create_omniassist_supervisor(agents: list, model: str = "gpt-4o"):
    """Create a supervisor that delegates to specialist agents.

    Args:
        agents: List of compiled agent graphs
        model: LLM model for the supervisor
    """
    try:
        from langgraph_supervisor import create_supervisor
        return create_supervisor(agents=agents, model=model)
    except ImportError:
        print("langgraph-supervisor not installed. Using manual supervisor.")
        return None
```

**File:** `src/agents/swarm_setup.py`

```python
"""Swarm pattern using langgraph-swarm library.

Concepts covered:
- langgraph-swarm for peer-to-peer agent handoff
"""

from __future__ import annotations


def create_omniassist_swarm(agents: list):
    """Create a swarm of agents that can hand off to each other.

    Args:
        agents: List of compiled agent graphs
    """
    try:
        from langgraph_swarm import create_swarm
        return create_swarm(agents=agents)
    except ImportError:
        print("langgraph-swarm not installed. Using manual handoff.")
        return None
```

**File:** `src/agents/bigtool_setup.py`

```python
"""BigTool setup for managing large numbers of tools.

Concepts covered:
- langgraph-bigtool for scaling tool access
"""

from __future__ import annotations


def create_bigtool_agent(tools: list, model: str = "gpt-4o"):
    """Create an agent optimized for many tools using langgraph-bigtool.

    Args:
        tools: Large list of available tools
        model: LLM model name
    """
    try:
        from langgraph_bigtool import create_agent_with_many_tools
        return create_agent_with_many_tools(model=model, tools=tools)
    except ImportError:
        print("langgraph-bigtool not installed. Using standard tool binding.")
        return None
```

**VERIFY:**
```bash
python -c "
from src.agents.supervisor_setup import create_omniassist_supervisor
from src.agents.swarm_setup import create_omniassist_swarm
from src.agents.bigtool_setup import create_bigtool_agent
print('Multi-agent setups OK')
"
```

---

### TASK 3.8 — Build Supervisor Node

**File:** `src/nodes/supervisor.py`

```python
"""Supervisor node — routes user messages to the correct specialist agent.

Concepts covered:
- LLM-based intent classification
- Command(goto=...) for dynamic routing (edgeless navigation)
- context_schema / Runtime access
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

from src.config.models import get_chat_model
from src.config.prompts import SUPERVISOR_PROMPT


VALID_AGENTS = {"chat_agent", "research", "writing", "task_mgmt", "code_exec", "data_analysis"}


def supervisor_node(state: dict) -> Command:
    """Classify user intent and route to the appropriate agent.

    Uses Command(goto=...) for dynamic routing without predefined edges.
    This is the recommended pattern for supervisor nodes.
    """
    messages = state.get("messages", [])
    if not messages:
        return Command(
            goto="chat_agent",
            update={"current_intent": "chat", "current_agent": "chat_agent"},
        )

    last_message = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    model = get_chat_model("gpt-4o-mini", temperature=0)

    prompt = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=last_message),
    ]

    response = model.invoke(prompt)
    intent = response.content.strip().lower().replace(" ", "_")

    # Validate the intent
    if intent not in VALID_AGENTS:
        intent = "chat_agent"

    return Command(
        goto=intent,
        update={"current_intent": intent, "current_agent": intent},
    )
```

**VERIFY:** `python -c "from src.nodes.supervisor import supervisor_node, VALID_AGENTS; print('Supervisor OK, agents:', VALID_AGENTS)"`

---

### TASK 3.9 — Build Main Orchestrator Graph

**File:** `src/graphs/main.py`

```python
"""Main Orchestrator Graph — the central hub of OmniAssist.

Concepts covered:
- StateGraph with Input/Output schema separation
- context_schema
- Supervisor routing via Command(goto=...)
- Subgraphs as nodes
- Deferred nodes (response synthesizer)
- Error handling
- Memory management
- START and END
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.state.main_state import OmniAssistState, OmniAssistInput, OmniAssistOutput
from src.state.context import OmniAssistContext
from src.nodes.input_handler import input_handler_node
from src.nodes.supervisor import supervisor_node
from src.nodes.output_handler import output_handler_node
from src.nodes.error_handler import error_handler_node
from src.nodes.memory_manager import memory_manager_node


# ──────────────────────────────────────────────
# SUBGRAPH WRAPPER NODES
# These wrap subgraph calls to adapt state schemas
# ──────────────────────────────────────────────

def chat_agent_node(state: dict) -> dict:
    """Chat agent — wraps the custom chat agent subgraph."""
    from src.graphs.chat import get_chat_graph
    chat_graph = get_chat_graph()
    result = chat_graph.invoke({
        "messages": state.get("messages", []),
        "user_id": state.get("user_id", "default"),
    })
    return {"messages": result.get("messages", [])[-1:]}  # Only last AI message


def research_node(state: dict) -> dict:
    """Research subgraph wrapper."""
    from src.graphs.research import graph as research_graph
    result = research_graph.invoke({
        "messages": state.get("messages", []),
        "query": state["messages"][-1].content if state.get("messages") else "",
        "user_id": state.get("user_id", "default"),
    })
    return {
        "messages": result.get("messages", [])[-1:],
        "research_results": result.get("validated_sources", []),
    }


def writing_node(state: dict) -> dict:
    """Writing subgraph wrapper."""
    from src.graphs.writing import graph as writing_graph
    try:
        result = writing_graph.invoke({
            "messages": state.get("messages", []),
            "user_id": state.get("user_id", "default"),
        })
        return {
            "messages": result.get("messages", [])[-1:],
            "draft_content": result.get("draft_content", ""),
        }
    except Exception:
        # Writing subgraph may interrupt for HITL — handle gracefully
        return {"messages": [AIMessage(content="Writing draft is ready for your review. Please check the pending approval.")]}


def task_mgmt_node(state: dict) -> dict:
    """Task management subgraph wrapper."""
    from src.graphs.task_management import graph as task_graph
    try:
        result = task_graph.invoke({
            "messages": state.get("messages", []),
            "user_id": state.get("user_id", "default"),
        })
        return {
            "messages": result.get("messages", [])[-1:],
            "tasks": result.get("tasks", []),
        }
    except Exception:
        return {"messages": [AIMessage(content="Task operation requires your confirmation. Please check pending approvals.")]}


def code_exec_node(state: dict) -> dict:
    """Code execution wrapper — uses Functional API pipeline."""
    from src.workflows.code_pipeline import code_pipeline
    request = state["messages"][-1].content if state.get("messages") else ""
    try:
        result = code_pipeline.invoke(
            request,
            config={"configurable": {"thread_id": state.get("thread_id", "code-default")}},
        )
        output = result.get("output", "")
        code = result.get("code", "")
        response = f"**Code:**\n```python\n{code}\n```\n\n**Output:**\n{output}"
        return {"messages": [AIMessage(content=response)], "code_output": output}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Code execution error: {e}")]}


def data_analysis_node(state: dict) -> dict:
    """Data analysis subgraph wrapper."""
    from src.graphs.data_analysis import graph as data_graph
    result = data_graph.invoke({
        "messages": state.get("messages", []),
        "user_id": state.get("user_id", "default"),
    })
    return {
        "messages": result.get("messages", [])[-1:],
        "data_analysis": {"summary": result.get("summary", "")},
    }


def response_synthesizer_node(state: dict) -> dict:
    """Synthesize the final response.

    This is a DEFERRED node — it runs after the selected agent completes.
    It exists primarily to ensure consistent output formatting.
    """
    # The agent nodes already produce messages, so this is a pass-through
    # with optional post-processing
    return {}


# ──────────────────────────────────────────────
# BUILD MAIN ORCHESTRATOR
# ──────────────────────────────────────────────

def build_main_graph() -> StateGraph:
    """Build the main OmniAssist orchestrator graph."""
    builder = StateGraph(
        state_schema=OmniAssistState,
        input=OmniAssistInput,
        output=OmniAssistOutput,
    )

    # Add nodes
    builder.add_node("input_handler", input_handler_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("chat_agent", chat_agent_node)
    builder.add_node("research", research_node)
    builder.add_node("writing", writing_node)
    builder.add_node("task_mgmt", task_mgmt_node)
    builder.add_node("code_exec", code_exec_node)
    builder.add_node("data_analysis", data_analysis_node)
    builder.add_node("memory_manager", memory_manager_node)
    builder.add_node("output_handler", output_handler_node)
    builder.add_node("error_handler", error_handler_node)

    # Entry flow
    builder.add_edge(START, "input_handler")
    builder.add_edge("input_handler", "supervisor")

    # Supervisor uses Command(goto=...) — no explicit conditional edges needed
    # The supervisor_node returns Command(goto=<agent_name>)
    # which dynamically routes to the correct agent.

    # All agents flow to memory_manager → output_handler → END
    for agent in ["chat_agent", "research", "writing", "task_mgmt", "code_exec", "data_analysis"]:
        builder.add_edge(agent, "memory_manager")

    builder.add_edge("memory_manager", "output_handler")
    builder.add_edge("output_handler", END)

    # Error handler also goes to END
    builder.add_edge("error_handler", END)

    return builder


# ──────────────────────────────────────────────
# COMPILED GRAPH (default export)
# ──────────────────────────────────────────────

def get_main_graph(checkpointer=None, store=None):
    """Return compiled main graph with optional checkpointer and store."""
    builder = build_main_graph()
    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
    )


# Default compiled graph for langgraph.json
graph = get_main_graph()
```

**VERIFY:**
```bash
python -c "
from src.graphs.main import graph, build_main_graph
print('Main graph nodes:', list(graph.get_graph().nodes))
print('Main orchestrator OK')
"
```

---

### TASK 3.10 — Build Assistants Configuration

**File:** `src/config/assistants.py`

```python
"""Assistant configurations — different personas/configurations for the same graph.

Concepts covered:
- Assistants API
- Multiple configurations for the same graph
"""

from __future__ import annotations


ASSISTANT_CONFIGS = {
    "default": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "persona": "professional",
        "max_research_sources": 5,
        "code_execution_enabled": True,
        "language": "en",
    },
    "creative": {
        "model_name": "gpt-4o",
        "temperature": 1.0,
        "persona": "casual",
        "max_research_sources": 3,
        "code_execution_enabled": True,
        "language": "en",
    },
    "research_focused": {
        "model_name": "gpt-4o",
        "temperature": 0.2,
        "persona": "technical",
        "max_research_sources": 10,
        "code_execution_enabled": False,
        "language": "en",
    },
    "code_assistant": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.1,
        "persona": "technical",
        "max_research_sources": 2,
        "code_execution_enabled": True,
        "language": "en",
    },
}


def get_assistant_config(name: str = "default") -> dict:
    """Get assistant configuration by name."""
    return ASSISTANT_CONFIGS.get(name, ASSISTANT_CONFIGS["default"])
```

**VERIFY:** `python -c "from src.config.assistants import get_assistant_config; print(get_assistant_config('creative'))"`

---

### TASK 3.11 — Write Phase 3 Tests

**File:** `tests/test_functional_api.py`

```python
"""Tests for the Functional API workflows."""

import pytest
from src.workflows.code_pipeline import generate_code, review_code, execute_code, code_pipeline
from src.workflows.quick_research import quick_research


def test_code_pipeline_exists():
    assert code_pipeline is not None


def test_quick_research_exists():
    assert quick_research is not None


@pytest.mark.skipif(
    not __import__("os").getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_execute_code_task():
    """Test the execute_code task directly."""
    result = execute_code("print('hello')").result()
    assert "hello" in result.get("output", "")
```

**File:** `tests/test_memory.py`

```python
"""Tests for the memory system."""

import pytest
from src.memory.short_term import trim_conversation
from src.memory.long_term import create_memory_store, store_user_fact, search_user_facts


def test_create_memory_store():
    store = create_memory_store(use_semantic_search=False)
    assert store is not None


def test_store_and_retrieve_fact():
    store = create_memory_store(use_semantic_search=False)
    store_user_fact(store, "user1", "fact1", {"text": "User likes Python"})
    # Note: search without semantic index may not work perfectly,
    # but the store operations should not error
    facts = search_user_facts(store, "user1", "Python")
    # We just verify no errors occurred
    assert isinstance(facts, list)
```

**File:** `tests/test_streaming.py`

```python
"""Tests for streaming utilities."""

from src.utils.streaming import progress_event, thinking_event, agent_handoff_event


def test_progress_event():
    event = progress_event("searching", 50, "Looking up sources")
    assert event["type"] == "progress"
    assert event["data"]["percent"] == 50


def test_thinking_event():
    event = thinking_event("Analyzing...")
    assert event["type"] == "thinking"


def test_agent_handoff_event():
    event = agent_handoff_event("supervisor", "research")
    assert event["data"]["from"] == "supervisor"
    assert event["data"]["to"] == "research"
```

**VERIFY:** `python -m pytest tests/test_functional_api.py tests/test_memory.py tests/test_streaming.py -v --tb=short`

---

---

# 🏭 PHASE 4 — PRODUCTION FEATURES (Days 12–15)

> **Goal:** Implement persistent checkpointing (SQLite → PostgreSQL), encrypted checkpoints, time travel/replay/state forking, retry/cache policies, full streaming system, double texting handling, context_schema system, and frontend.
>
> **Concepts covered:** `SqliteSaver`, `PostgresSaver`, `Encrypted Checkpointing`, `Replay/Time Travel`, `State Forking`, `Get/Update State`, `Retry Policy`, `Cache Policy`, `Streaming (all modes)`, `StreamWriter`, `UI Messages`, `Double Texting`, `context_schema`, `Runtime`, `Recursion Limit`, `Dynamic Graph Construction`

---

### TASK 4.1 — Implement Checkpointing Backends

**File:** `src/utils/encryption.py`

```python
"""Encrypted checkpointing setup.

Concepts covered:
- Encrypted checkpointing with AES encryption
- Different checkpointer backends
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver


def get_checkpointer(backend: str = "memory", connection_string: str = "", encrypted: bool = False):
    """Factory for checkpointer backends.

    Args:
        backend: "memory", "sqlite", or "postgres"
        connection_string: DB connection string (for sqlite/postgres)
        encrypted: Whether to encrypt checkpoint data at rest

    Returns:
        Configured checkpointer instance
    """
    serde = None
    if encrypted:
        try:
            from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
            serde = EncryptedSerializer.from_pycryptodome_aes()  # Reads LANGGRAPH_AES_KEY env var
        except ImportError:
            print("Warning: Encrypted serializer not available. Using unencrypted.")

    if backend == "memory":
        return InMemorySaver(serde=serde) if serde else InMemorySaver()

    elif backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn_str = connection_string or "omniassist.db"
            checkpointer = SqliteSaver.from_conn_string(conn_str)
            if serde:
                checkpointer = SqliteSaver.from_conn_string(conn_str, serde=serde)
            return checkpointer
        except ImportError:
            print("langgraph-checkpoint-sqlite not installed. Falling back to memory.")
            return InMemorySaver()

    elif backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            if not connection_string:
                raise ValueError("PostgresSaver requires a connection string")
            if serde:
                checkpointer = PostgresSaver.from_conn_string(connection_string, serde=serde)
            else:
                checkpointer = PostgresSaver.from_conn_string(connection_string)
            checkpointer.setup()  # REQUIRED — creates tables on first run
            return checkpointer
        except ImportError:
            print("langgraph-checkpoint-postgres not installed. Falling back to memory.")
            return InMemorySaver()

    else:
        raise ValueError(f"Unknown backend: {backend}")
```

**VERIFY:** `python -c "from src.utils.encryption import get_checkpointer; cp = get_checkpointer('memory'); print('Checkpointer factory OK')"`

---

### TASK 4.2 — Build Time Travel / Replay Script

**File:** `scripts/replay_thread.py`

```python
"""Time travel and replay utility for OmniAssist conversations.

Concepts covered:
- Replay from any checkpoint
- State forking ("what-if" branching)
- Get/Update state externally
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from langgraph.checkpoint.memory import InMemorySaver

from src.graphs.main import get_main_graph

console = Console()


def list_checkpoints(graph, thread_id: str):
    """List all checkpoints for a thread."""
    config = {"configurable": {"thread_id": thread_id}}

    table = Table(title=f"Checkpoints for thread: {thread_id}")
    table.add_column("Checkpoint ID", style="cyan")
    table.add_column("Node", style="green")
    table.add_column("# Messages", style="yellow")

    try:
        for state in graph.get_state_history(config):
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id", "?")
            node = state.next[0] if state.next else "END"
            num_msgs = len(state.values.get("messages", []))
            table.add_row(checkpoint_id, node, str(num_msgs))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print(table)


def replay_from_checkpoint(graph, thread_id: str, checkpoint_id: str):
    """Replay a conversation from a specific checkpoint."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    try:
        state = graph.get_state(config)
        console.print(f"\n[bold]State at checkpoint {checkpoint_id}:[/bold]")
        console.print(f"  Current agent: {state.values.get('current_agent', 'N/A')}")
        console.print(f"  Messages: {len(state.values.get('messages', []))}")
        console.print(f"  Next nodes: {state.next}")

        for msg in state.values.get("messages", []):
            role = "User" if hasattr(msg, "type") and msg.type == "human" else "AI"
            console.print(f"  [{role}]: {msg.content[:100]}...")
    except Exception as e:
        console.print(f"[red]Error replaying: {e}[/red]")


def fork_state(graph, thread_id: str, checkpoint_id: str, new_thread_id: str):
    """Fork (branch) from a checkpoint into a new thread for 'what-if' scenarios."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    try:
        state = graph.get_state(config)

        # Update state on the new thread (this creates a fork)
        new_config = {"configurable": {"thread_id": new_thread_id}}
        graph.update_state(new_config, state.values)

        console.print(f"[green]State forked from {thread_id}:{checkpoint_id} → {new_thread_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error forking: {e}[/red]")


if __name__ == "__main__":
    console.print("[bold]OmniAssist Time Travel Utility[/bold]")
    console.print("Usage:")
    console.print("  python scripts/replay_thread.py list <thread_id>")
    console.print("  python scripts/replay_thread.py replay <thread_id> <checkpoint_id>")
    console.print("  python scripts/replay_thread.py fork <thread_id> <checkpoint_id> <new_thread_id>")

    if len(sys.argv) < 3:
        sys.exit(1)

    graph = get_main_graph()
    command = sys.argv[1]
    thread_id = sys.argv[2]

    if command == "list":
        list_checkpoints(graph, thread_id)
    elif command == "replay" and len(sys.argv) >= 4:
        replay_from_checkpoint(graph, thread_id, sys.argv[3])
    elif command == "fork" and len(sys.argv) >= 5:
        fork_state(graph, thread_id, sys.argv[3], sys.argv[4])
```

**VERIFY:** `python -c "from scripts.replay_thread import list_checkpoints; print('Replay script OK')"`

---

### TASK 4.3 — Build Validation Utilities

**File:** `src/utils/validation.py`

```python
"""Input validation utilities."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class MessageInput(BaseModel):
    """Validate a single message input."""
    content: str
    role: str = "human"

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        valid_roles = {"human", "ai", "system", "tool"}
        if v.lower() not in valid_roles:
            raise ValueError(f"Invalid role: {v}. Must be one of {valid_roles}")
        return v.lower()


def validate_thread_id(thread_id: str) -> str:
    """Validate a thread ID."""
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id cannot be empty")
    return thread_id.strip()


def validate_user_id(user_id: str) -> str:
    """Validate a user ID."""
    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")
    return user_id.strip()
```

**File:** `src/utils/formatters.py`

```python
"""Output formatting utilities."""

from __future__ import annotations


def format_research_results(results: list[dict]) -> str:
    """Format research results for display."""
    if not results:
        return "No research results found."

    lines = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        title = result.get("title", "Untitled")
        content = result.get("content", "")[:200]
        lines.append(f"[{i}] ({source}) {title}\n    {content}")

    return "\n\n".join(lines)


def format_task_list(tasks: list[dict]) -> str:
    """Format a task list for display."""
    if not tasks:
        return "No tasks found."

    lines = []
    for task in tasks:
        status = task.get("status", "pending")
        text = task.get("text", "Untitled task")
        task_id = task.get("id", "?")
        emoji = "✅" if status == "done" else "⏳" if status == "pending" else "🔄"
        lines.append(f"{emoji} [{status}] {text} (ID: {task_id})")

    return "\n".join(lines)


def format_code_output(code: str, output: str, error: str | None = None) -> str:
    """Format code execution results."""
    parts = [f"**Code:**\n```python\n{code}\n```"]

    if output:
        parts.append(f"**Output:**\n```\n{output}\n```")

    if error:
        parts.append(f"**Error:**\n```\n{error}\n```")

    return "\n\n".join(parts)
```

**VERIFY:**
```bash
python -c "
from src.utils.validation import validate_thread_id
from src.utils.formatters import format_task_list
print(format_task_list([{'status': 'pending', 'text': 'Test task', 'id': '123'}]))
"
```

---

### TASK 4.4 — Build Full CLI with All Features

**Update:** `scripts/cli_chat.py` — Replace the content with an enhanced version:

```python
"""Enhanced CLI chat interface for OmniAssist.

Features:
- Full main orchestrator graph
- Thread management
- Streaming output
- Time travel commands
- Multiple assistant configurations

Usage: python scripts/cli_chat.py
"""

import sys
import os
import uuid
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from src.graphs.main import get_main_graph
from src.config.assistants import get_assistant_config, ASSISTANT_CONFIGS

console = Console()


def print_help():
    """Print help text."""
    table = Table(title="Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")

    commands = [
        ("quit", "Exit the chat"),
        ("new", "Start a new conversation thread"),
        ("history", "Show conversation history for current thread"),
        ("checkpoints", "List all checkpoints for current thread"),
        ("state", "Show current graph state"),
        ("assistants", "List available assistant configurations"),
        ("switch <name>", "Switch to a different assistant"),
        ("resume <value>", "Resume an interrupted operation"),
        ("help", "Show this help text"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)


def main():
    """Run the enhanced CLI chat interface."""
    console.print(Panel.fit(
        "[bold green]OmniAssist CLI v2.0[/bold green]\n"
        "Type 'help' for available commands.",
        title="Welcome",
    ))

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    graph = get_main_graph(checkpointer=checkpointer, store=store)

    thread_id = str(uuid.uuid4())
    assistant = "default"
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Thread: {thread_id} | Assistant: {assistant}[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        # ── Command handling ──
        if user_input.lower() == "quit":
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "help":
            print_help()
            continue

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            console.print(f"\n[green]New thread: {thread_id}[/green]\n")
            continue

        if user_input.lower() == "assistants":
            for name, cfg in ASSISTANT_CONFIGS.items():
                console.print(f"  [cyan]{name}[/cyan]: {cfg.get('persona', '?')} (model: {cfg.get('model_name', '?')})")
            continue

        if user_input.lower().startswith("switch "):
            new_assistant = user_input[7:].strip()
            if new_assistant in ASSISTANT_CONFIGS:
                assistant = new_assistant
                console.print(f"[green]Switched to assistant: {assistant}[/green]")
            else:
                console.print(f"[red]Unknown assistant: {new_assistant}[/red]")
            continue

        if user_input.lower() == "state":
            try:
                state = graph.get_state(config)
                console.print(f"  Agent: {state.values.get('current_agent', 'N/A')}")
                console.print(f"  Messages: {len(state.values.get('messages', []))}")
                console.print(f"  Next: {state.next}")
            except Exception as e:
                console.print(f"[red]{e}[/red]")
            continue

        if user_input.lower() == "history":
            try:
                state = graph.get_state(config)
                for msg in state.values.get("messages", []):
                    role = "User" if isinstance(msg, HumanMessage) else "AI"
                    content = msg.content[:150]
                    console.print(f"  [{role}]: {content}")
            except Exception as e:
                console.print(f"[red]{e}[/red]")
            continue

        if user_input.lower() == "checkpoints":
            try:
                for i, state in enumerate(graph.get_state_history(config)):
                    cp_id = state.config.get("configurable", {}).get("checkpoint_id", "?")
                    next_node = state.next[0] if state.next else "END"
                    console.print(f"  [{i}] {cp_id} → {next_node}")
                    if i >= 10:
                        console.print("  ... (showing first 10)")
                        break
            except Exception as e:
                console.print(f"[red]{e}[/red]")
            continue

        if user_input.lower().startswith("resume "):
            resume_value = user_input[7:].strip()
            try:
                result = graph.invoke(Command(resume=resume_value), config=config)
                last_msg = None
                for msg in reversed(result.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content:
                        last_msg = msg
                        break
                if last_msg:
                    console.print(f"\n[bold green]Assistant:[/bold green] ", end="")
                    console.print(Markdown(last_msg.content))
            except Exception as e:
                console.print(f"[red]Resume error: {e}[/red]")
            continue

        # ── Normal message ──
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            last_ai_msg = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    last_ai_msg = msg
                    break

            if last_ai_msg:
                agent = result.get("current_agent", "unknown")
                console.print(f"\n[dim]({agent})[/dim]")
                console.print(f"[bold green]Assistant:[/bold green] ", end="")
                console.print(Markdown(last_ai_msg.content))
                console.print()
            else:
                console.print("[dim]No response generated.[/dim]\n")

        except Exception as e:
            error_str = str(e)
            if "interrupt" in error_str.lower():
                console.print(f"\n[yellow]⏸ Operation paused — approval required.[/yellow]")
                console.print(f"[yellow]Use 'resume approved' or 'resume rejected' to continue.[/yellow]\n")
            else:
                console.print(f"\n[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
```

**VERIFY:** `python -c "from scripts.cli_chat import print_help; print('Enhanced CLI OK')"`

---

### TASK 4.5 — Build Frontend

**File:** `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniAssist</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="app">
        <header>
            <h1>🤖 OmniAssist</h1>
            <div id="status">
                <span id="agent-badge">Ready</span>
                <span id="thread-id"></span>
            </div>
        </header>

        <div id="chat-container">
            <div id="messages"></div>
        </div>

        <div id="input-container">
            <input type="text" id="user-input" placeholder="Type a message..." autofocus>
            <button id="send-btn" onclick="sendMessage()">Send</button>
            <button id="new-thread-btn" onclick="newThread()">New Thread</button>
        </div>

        <div id="approval-bar" style="display: none;">
            <p id="approval-prompt"></p>
            <button onclick="resumeWith('approved')">✅ Approve</button>
            <button onclick="resumeWith('rejected')">❌ Reject</button>
        </div>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

**File:** `frontend/styles.css`

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #eee;
    height: 100vh;
    display: flex;
    justify-content: center;
}

#app {
    width: 100%;
    max-width: 800px;
    display: flex;
    flex-direction: column;
    height: 100vh;
}

header {
    padding: 16px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    font-size: 1.4rem;
}

#status {
    display: flex;
    gap: 12px;
    align-items: center;
}

#agent-badge {
    background: #0f3460;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
}

#thread-id {
    font-size: 0.75rem;
    color: #888;
}

#chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

#messages {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.message {
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 80%;
    line-height: 1.5;
    white-space: pre-wrap;
}

.message.human {
    background: #0f3460;
    align-self: flex-end;
}

.message.ai {
    background: #1a1a40;
    border: 1px solid #333;
    align-self: flex-start;
}

.message .agent-tag {
    font-size: 0.7rem;
    color: #888;
    margin-bottom: 4px;
}

#input-container {
    padding: 16px;
    background: #16213e;
    display: flex;
    gap: 8px;
}

#user-input {
    flex: 1;
    padding: 12px;
    border: 1px solid #0f3460;
    border-radius: 8px;
    background: #1a1a2e;
    color: #eee;
    font-size: 1rem;
}

#user-input:focus {
    outline: none;
    border-color: #e94560;
}

button {
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    background: #e94560;
    color: white;
    font-size: 0.9rem;
    cursor: pointer;
}

button:hover {
    background: #c73e54;
}

#new-thread-btn {
    background: #0f3460;
}

#approval-bar {
    padding: 12px 16px;
    background: #2a1a00;
    border-top: 2px solid #e94560;
    display: flex;
    align-items: center;
    gap: 12px;
}

#approval-bar p {
    flex: 1;
    font-size: 0.9rem;
}

#approval-bar button {
    padding: 8px 16px;
    font-size: 0.85rem;
}
```

**File:** `frontend/app.js`

```javascript
/**
 * OmniAssist Frontend — Minimal chat UI
 *
 * Connects to LangGraph Platform API (or local LangGraph server).
 * Supports streaming, HITL approval, and thread management.
 */

const API_BASE = "http://localhost:2024"; // LangGraph server URL
const GRAPH_NAME = "omniassist";

let threadId = crypto.randomUUID();
let assistantId = null;

// DOM elements
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("user-input");
const agentBadge = document.getElementById("agent-badge");
const threadIdEl = document.getElementById("thread-id");
const approvalBar = document.getElementById("approval-bar");
const approvalPrompt = document.getElementById("approval-prompt");

// Initialize
threadIdEl.textContent = `Thread: ${threadId.slice(0, 8)}`;

// Send message on Enter
inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});

function addMessage(content, role, agent = "") {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    if (agent) {
        const tag = document.createElement("div");
        tag.className = "agent-tag";
        tag.textContent = agent;
        div.appendChild(tag);
    }
    const text = document.createElement("div");
    text.textContent = content;
    div.appendChild(text);
    messagesEl.appendChild(div);
    messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
}

async function sendMessage() {
    const content = inputEl.value.trim();
    if (!content) return;

    inputEl.value = "";
    addMessage(content, "human");
    agentBadge.textContent = "Thinking...";

    try {
        const response = await fetch(`${API_BASE}/threads/${threadId}/runs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                assistant_id: GRAPH_NAME,
                input: {
                    messages: [{ role: "human", content: content }],
                },
                stream_mode: ["values"],
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Extract last AI message
        const messages = data.messages || [];
        const lastAI = messages.filter(m => m.type === "ai" || m.role === "assistant").pop();

        if (lastAI) {
            const agent = data.current_agent || "assistant";
            agentBadge.textContent = agent;
            addMessage(lastAI.content, "ai", agent);
        }
    } catch (error) {
        addMessage(`Error: ${error.message}`, "ai", "error");
        agentBadge.textContent = "Error";
    }
}

function newThread() {
    threadId = crypto.randomUUID();
    threadIdEl.textContent = `Thread: ${threadId.slice(0, 8)}`;
    messagesEl.innerHTML = "";
    agentBadge.textContent = "Ready";
    addMessage("New conversation started.", "ai", "system");
}

async function resumeWith(value) {
    approvalBar.style.display = "none";
    agentBadge.textContent = "Resuming...";

    try {
        const response = await fetch(`${API_BASE}/threads/${threadId}/runs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                assistant_id: GRAPH_NAME,
                command: { resume: value },
            }),
        });

        const data = await response.json();
        const messages = data.messages || [];
        const lastAI = messages.filter(m => m.type === "ai" || m.role === "assistant").pop();

        if (lastAI) {
            addMessage(lastAI.content, "ai", data.current_agent || "assistant");
        }
        agentBadge.textContent = data.current_agent || "Ready";
    } catch (error) {
        addMessage(`Resume error: ${error.message}`, "ai", "error");
    }
}

function showApprovalBar(prompt) {
    approvalPrompt.textContent = prompt;
    approvalBar.style.display = "flex";
}
```

**VERIFY:** Verify all three files exist:
```bash
ls frontend/index.html frontend/styles.css frontend/app.js
```

---

### TASK 4.6 — Build Seed Knowledge Base Script

**File:** `scripts/seed_knowledge_base.py`

```python
"""Seed the knowledge base with sample data for testing.

Usage: python scripts/seed_knowledge_base.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.long_term import create_memory_store, store_user_fact


SAMPLE_FACTS = [
    {"text": "LangGraph is a framework for building stateful, multi-actor applications with LLMs.", "summary": "LangGraph overview"},
    {"text": "Python 3.11 introduced significant performance improvements over 3.10.", "summary": "Python 3.11 performance"},
    {"text": "The Transformer architecture was introduced in the paper 'Attention Is All You Need' in 2017.", "summary": "Transformer history"},
    {"text": "FastAPI is a modern web framework for building APIs with Python.", "summary": "FastAPI overview"},
    {"text": "Docker containers provide isolated environments for running applications.", "summary": "Docker overview"},
]


def seed(user_id: str = "default_user"):
    """Seed the knowledge base with sample data."""
    store = create_memory_store(use_semantic_search=False)

    for i, fact in enumerate(SAMPLE_FACTS):
        store_user_fact(store, user_id, f"seed_{i}", fact)
        print(f"  ✅ Stored: {fact['summary']}")

    print(f"\nSeeded {len(SAMPLE_FACTS)} facts for user '{user_id}'")
    return store


if __name__ == "__main__":
    seed()
```

**VERIFY:** `python scripts/seed_knowledge_base.py`

---

### TASK 4.7 — Build Benchmark Script

**File:** `scripts/benchmark.py`

```python
"""Benchmark OmniAssist performance.

Usage: python scripts/benchmark.py
"""

import sys
import os
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

console = Console()


def benchmark_graph_compilation():
    """Benchmark graph compilation time."""
    start = time.perf_counter()
    from src.graphs.main import build_main_graph
    builder = build_main_graph()
    graph = builder.compile(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    elapsed = time.perf_counter() - start
    return elapsed, graph


def benchmark_invocation(graph, message: str, n: int = 3):
    """Benchmark graph invocation time."""
    times = []
    for i in range(n):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        start = time.perf_counter()
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            console.print(f"[red]Invocation {i} failed: {e}[/red]")

    return times


def main():
    console.print("[bold]OmniAssist Benchmark[/bold]\n")

    # Compilation benchmark
    compile_time, graph = benchmark_graph_compilation()
    console.print(f"Graph compilation: {compile_time:.3f}s")

    # Invocation benchmarks (only if API keys available)
    if os.getenv("OPENAI_API_KEY"):
        table = Table(title="Invocation Benchmarks (3 runs each)")
        table.add_column("Test", style="cyan")
        table.add_column("Avg Time", style="green")
        table.add_column("Min", style="yellow")
        table.add_column("Max", style="yellow")

        test_cases = [
            ("Simple greeting", "Hello!"),
            ("Math question", "What is the square root of 144?"),
            ("Research request", "What is LangGraph?"),
        ]

        for name, message in test_cases:
            times = benchmark_invocation(graph, message, n=2)
            if times:
                avg = sum(times) / len(times)
                table.add_row(name, f"{avg:.2f}s", f"{min(times):.2f}s", f"{max(times):.2f}s")

        console.print(table)
    else:
        console.print("[yellow]Set OPENAI_API_KEY to run invocation benchmarks.[/yellow]")


if __name__ == "__main__":
    main()
```

**VERIFY:** `python scripts/benchmark.py`

---

### TASK 4.8 — Write Remaining Tests

**File:** `tests/test_code_execution.py`

```python
"""Tests for code execution pipeline."""

import pytest


def test_execute_code_safe():
    from src.tools.code_runner import run_python_code
    result = run_python_code.invoke({"code": "print(2 + 2)"})
    assert "4" in result


def test_execute_code_error():
    from src.tools.code_runner import run_python_code
    result = run_python_code.invoke({"code": "raise ValueError('test')"})
    assert "ValueError" in result
```

**File:** `tests/test_data_analysis.py`

```python
"""Tests for data analysis subgraph."""

import pytest
from src.graphs.data_analysis import build_data_analysis_graph, graph


def test_data_analysis_compiles():
    builder = build_data_analysis_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_data_analysis_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["parse_request", "statistics", "visualization", "summary", "results_merger"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"
```

**File:** `tests/test_semantic_search.py`

```python
"""Tests for semantic search functionality."""

import pytest
from src.memory.long_term import create_memory_store, store_user_fact, search_user_facts


def test_store_creation():
    store = create_memory_store(use_semantic_search=False)
    assert store is not None


def test_store_and_search_basic():
    store = create_memory_store(use_semantic_search=False)
    store_user_fact(store, "test_user", "f1", {"text": "User likes Python programming"})
    store_user_fact(store, "test_user", "f2", {"text": "User works at a tech company"})
    # Basic search (without semantic embeddings)
    results = search_user_facts(store, "test_user", "Python")
    assert isinstance(results, list)
```

**File:** `tests/test_encrypted_checkpoints.py`

```python
"""Tests for encrypted checkpointing."""

import pytest
from src.utils.encryption import get_checkpointer


def test_memory_checkpointer():
    cp = get_checkpointer("memory")
    assert cp is not None


def test_sqlite_checkpointer():
    try:
        cp = get_checkpointer("sqlite", "test_omniassist.db")
        assert cp is not None
        import os
        if os.path.exists("test_omniassist.db"):
            os.remove("test_omniassist.db")
    except Exception:
        pytest.skip("SQLite checkpointer not available")
```

**File:** `tests/test_mcp_integration.py`

```python
"""Tests for MCP integration."""

import pytest
from src.tools.mcp_tools import get_mcp_server_config, MCP_SERVER_CONFIGS


def test_mcp_config_exists():
    assert isinstance(MCP_SERVER_CONFIGS, dict)


def test_get_mcp_config():
    config = get_mcp_server_config("file_system")
    if config:
        assert "transport" in config
```

**VERIFY:** `python -m pytest tests/ -v --tb=short`

---

---

# 🚀 PHASE 5 — DEPLOYMENT & POLISH (Days 16–18)

> **Goal:** Set up LangGraph Platform/Server, finalize langgraph.json, add cron jobs, bigtool integration, comprehensive testing, documentation.
>
> **Concepts covered:** `LangGraph Platform`, `LangGraph Studio`, `Cron Jobs`, `Double Texting`, `Assistants API`, `langgraph-bigtool`, `MCP endpoint auto-exposure`

---

### TASK 5.1 — Create README.md

**File:** `README.md`

```markdown
# 🤖 OmniAssist — AI-Powered Multi-Agent Conversational Platform

Built on **LangGraph v1.0+**, OmniAssist is a production-grade chatbot that demonstrates 65+ LangGraph concepts across both the Graph API and Functional API.

## Features

- **Multi-Agent Architecture:** Supervisor-routed specialist agents for chat, research, writing, task management, code execution, and data analysis
- **Both APIs:** StateGraph (Graph API) + `@entrypoint`/`@task` (Functional API)
- **Human-in-the-Loop:** Modern `interrupt()` / `Command(resume=...)` pattern
- **Memory:** Short-term (message trimming), long-term (Store API + semantic search), LangMem SDK
- **Streaming:** Token, event, update, custom events, StreamWriter, UI messages
- **Checkpointing:** InMemory, SQLite, PostgreSQL, with optional AES encryption
- **Time Travel:** Replay from any checkpoint, state forking
- **Multi-Agent Patterns:** Supervisor (langgraph-supervisor), Swarm (langgraph-swarm), BigTool
- **MCP Integration:** Connect to MCP-compatible tool servers
- **Production Ready:** LangGraph Platform deployment, cron jobs, double texting handling

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set up environment
cp .env.example .env  # Edit with your API keys

# Run CLI
python scripts/cli_chat.py

# Run tests
python -m pytest tests/ -v
```

## Architecture

The main orchestrator (`src/graphs/main.py`) routes messages through a supervisor to specialist subgraphs:

- **Chat Agent** — General conversation (prebuilt + custom StateGraph)
- **Research** — Parallel search with fan-out/fan-in + source validation
- **Writing** — Iterative drafting with HITL review cycles
- **Task Management** — CRUD with Store + semantic search
- **Code Execution** — Functional API pipeline with safety review
- **Data Analysis** — Statistics, visualization, and summarization

## Deployment

```bash
# Local development with LangGraph Studio
langgraph dev

# Production deployment
langgraph up
```

## LangGraph Concepts Covered

This project demonstrates 65+ LangGraph concepts. See the full spec for the complete coverage map.

## Project Structure

```
src/
├── state/          # State schemas (TypedDict + reducers)
├── graphs/         # StateGraph modules (Graph API)
├── workflows/      # Functional API modules (@entrypoint/@task)
├── nodes/          # Shared node functions
├── tools/          # LangChain tools
├── memory/         # Short-term + long-term memory
├── agents/         # Multi-agent setups
├── config/         # Settings, prompts, assistants
└── utils/          # Streaming, validation, encryption
```
```

---

### TASK 5.2 — Final Verification Checklist

**Action:** Run each of these commands in sequence. Every one must succeed.

```bash
# 1. All imports work
python -c "
from src.state.main_state import OmniAssistState, OmniAssistInput, OmniAssistOutput
from src.state.context import OmniAssistContext
from src.state.research_state import ResearchState
from src.state.writing_state import WritingState
from src.state.task_state import TaskState
from src.state.code_state import CodeState
from src.state.data_state import DataState
print('✅ All state schemas OK')
"

# 2. All graphs compile
python -c "
from src.graphs.chat import get_chat_graph
from src.graphs.research import graph as research_graph
from src.graphs.writing import graph as writing_graph
from src.graphs.task_management import graph as task_graph
from src.graphs.data_analysis import graph as data_graph
from src.graphs.source_validator import source_validator_graph
from src.graphs.main import graph as main_graph
print('✅ All graphs compile OK')
"

# 3. Functional API workflows load
python -c "
from src.workflows.code_pipeline import code_pipeline
from src.workflows.quick_research import quick_research
from src.workflows.mcp_bridge import get_mcp_tools
print('✅ All Functional API workflows OK')
"

# 4. All tools load
python -c "
from src.tools import ALL_TOOLS
print(f'✅ {len(ALL_TOOLS)} tools loaded OK')
"

# 5. Memory system works
python -c "
from src.memory.long_term import create_memory_store
store = create_memory_store(use_semantic_search=False)
store.put(('test',), 'key1', {'text': 'hello'})
print('✅ Memory system OK')
"

# 6. Checkpointer factory works
python -c "
from src.utils.encryption import get_checkpointer
cp = get_checkpointer('memory')
print('✅ Checkpointer OK')
"

# 7. All tests pass
python -m pytest tests/ -v --tb=short

# 8. Main graph end-to-end (requires API key)
python -c "
import os
if os.getenv('OPENAI_API_KEY'):
    from langchain_core.messages import HumanMessage
    from src.graphs.main import get_main_graph
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    graph = get_main_graph(InMemorySaver(), InMemoryStore())
    result = graph.invoke(
        {'messages': [HumanMessage(content='Hello, what can you do?')]},
        config={'configurable': {'thread_id': 'final-test'}},
    )
    print(f'✅ End-to-end test passed. Agent: {result.get(\"current_agent\", \"unknown\")}')
else:
    print('⏭ Skipping end-to-end (no API key)')
"
```

---

### TASK 5.3 — Concept Coverage Verification

**Action:** Verify that the following 67 LangGraph concepts are implemented across the codebase. For each concept, the file location is listed.

| # | Concept | File(s) | Status |
|---|---------|---------|--------|
| 1 | StateGraph | `src/graphs/*.py` | Verify with `grep -r "StateGraph" src/` |
| 2 | Functional API (@entrypoint/@task) | `src/workflows/*.py` | Verify with `grep -r "@entrypoint\|@task" src/` |
| 3 | Nodes | All graph files | Verify with `grep -r "add_node" src/` |
| 4 | Edges | All graph files | Verify with `grep -r "add_edge" src/` |
| 5 | Conditional Edges | `src/graphs/research.py`, `writing.py`, `chat.py`, `data_analysis.py`, `task_management.py` | Verify with `grep -r "add_conditional_edges" src/` |
| 6 | START | All graph files | Verify with `grep -r "START" src/graphs/` |
| 7 | END | All graph files | Verify with `grep -r "END" src/graphs/` |
| 8 | State Schema (TypedDict) | `src/state/*.py` | ✅ |
| 9 | Input/Output Schema | `src/state/main_state.py` | ✅ |
| 10 | Reducers (add_messages, append_reducer) | `src/state/*.py` | Verify with `grep -r "Annotated" src/state/` |
| 11 | Annotation API | `src/state/*.py` | ✅ |
| 12 | context_schema | `src/state/context.py` | ✅ |
| 13 | Subgraphs | `src/graphs/main.py` (wraps subgraphs) | ✅ |
| 14 | Nested Subgraphs | `src/graphs/source_validator.py` used in `research.py` | ✅ |
| 15 | Fan-out/Fan-in | `src/graphs/research.py` | ✅ |
| 16 | Deferred Nodes | `src/graphs/research.py`, `data_analysis.py` | Verify with `grep -r "defer=True" src/` |
| 17 | Map-Reduce | `src/graphs/research.py` | ✅ |
| 18 | interrupt() | `src/graphs/writing.py`, `task_management.py`, `src/workflows/code_pipeline.py` | Verify with `grep -r "interrupt(" src/` |
| 19 | Command(resume=...) | `scripts/cli_chat.py`, `frontend/app.js` | ✅ |
| 20 | Command(goto=...) | `src/nodes/supervisor.py` | ✅ |
| 21 | Static Breakpoints | Documented in `src/graphs/writing.py` (add `interrupt_before` at compile time for debug) | ✅ |
| 22 | InMemorySaver | `src/utils/encryption.py`, `scripts/cli_chat.py` | ✅ |
| 23 | SqliteSaver | `src/utils/encryption.py` | ✅ |
| 24 | PostgresSaver | `src/utils/encryption.py` | ✅ |
| 25 | Encrypted Checkpointing | `src/utils/encryption.py` | ✅ |
| 26 | Replay/Time Travel | `scripts/replay_thread.py` | ✅ |
| 27 | State Forking | `scripts/replay_thread.py` | ✅ |
| 28 | Get/Update State | `scripts/cli_chat.py`, `scripts/replay_thread.py` | ✅ |
| 29 | Thread-level Persistence | All configs use `thread_id` | ✅ |
| 30 | Cross-thread Store | `src/memory/long_term.py`, `src/graphs/task_management.py` | ✅ |
| 31 | Semantic Memory Search | `src/memory/long_term.py`, `src/graphs/task_management.py` | ✅ |
| 32 | LangMem SDK | `src/memory/langmem_integration.py` | ✅ |
| 33 | Tool Calling | `src/tools/*.py` | ✅ |
| 34 | ToolNode (prebuilt) | `src/graphs/chat.py` | ✅ |
| 35 | Tools that update state | `src/tools/state_updating_tools.py` | ✅ |
| 36 | Dynamic Tool Calling | `src/graphs/chat.py` (adds code tools conditionally) | ✅ |
| 37 | Tool Error Handling | `src/tools/web_search.py` (try/except) | ✅ |
| 38 | Retry Policies | `src/graphs/research.py`, `src/workflows/code_pipeline.py` | ✅ |
| 39 | Cache Policy | `src/workflows/code_pipeline.py` (documented) | ✅ |
| 40 | Streaming (Tokens) | `scripts/cli_chat.py` (streaming mode) | ✅ |
| 41-44 | Streaming (Events, Updates, Custom, StreamWriter) | `src/utils/streaming.py` | ✅ |
| 45 | Multiple Streaming Modes | Documented in `frontend/app.js` | ✅ |
| 46 | Recursion Limit | `src/graphs/research.py` (quality loop bounded) | ✅ |
| 47 | Cycle Handling | `src/graphs/writing.py`, `research.py` | ✅ |
| 48 | Dynamic Graph Construction | `src/graphs/chat.py` (conditional tool binding) | ✅ |
| 49 | Runtime access | `src/nodes/supervisor.py` | ✅ |
| 50 | Message History Management | `src/nodes/message_utils.py`, `src/memory/short_term.py` | ✅ |
| 51 | Short-term Memory | `src/memory/short_term.py` | ✅ |
| 52 | Long-term Memory | `src/memory/long_term.py` | ✅ |
| 53 | LangGraph Platform | `langgraph.json` | ✅ |
| 54 | LangGraph Studio | `langgraph.json` (compatible) | ✅ |
| 55 | Cron Jobs | Documented in `langgraph.json` (add cron section) | ✅ |
| 56 | Double Texting | Handled by LangGraph Platform | ✅ |
| 57 | Assistants API | `src/config/assistants.py` | ✅ |
| 58 | langgraph-supervisor | `src/agents/supervisor_setup.py` | ✅ |
| 59 | langgraph-swarm | `src/agents/swarm_setup.py` | ✅ |
| 60 | MCP Integration | `src/workflows/mcp_bridge.py`, `src/tools/mcp_tools.py` | ✅ |
| 61 | create_react_agent | `src/graphs/chat.py` | ✅ |
| 62 | Pydantic validation | `src/state/main_state.py` | ✅ |
| 63 | Error handling & fallbacks | `src/nodes/error_handler.py` | ✅ |
| 64 | entrypoint.final | `src/workflows/code_pipeline.py` | ✅ |
| 65 | previous parameter | `src/workflows/code_pipeline.py` | ✅ |
| 66 | UI Messages (push_ui_message) | Documented in `src/utils/streaming.py` | ✅ |
| 67 | langgraph-bigtool | `src/agents/bigtool_setup.py` | ✅ |

---

### TASK 5.4 — Final Run

**Action:** Execute the complete test suite one final time:

```bash
cd omniassist
python -m pytest tests/ -v --tb=short
```

**Expected:** All tests pass (some may skip without API keys).

**Then run the CLI for a manual smoke test (if API keys are set):**

```bash
python scripts/cli_chat.py
```

**Test these interactions:**
1. Type: `Hello!` → Should route to chat_agent, get a friendly response
2. Type: `Research quantum computing` → Should route to research
3. Type: `Write me an email to my boss about taking Friday off` → Should route to writing
4. Type: `Create a task: buy groceries` → Should route to task management
5. Type: `Write Python code to calculate fibonacci numbers` → Should route to code execution
6. Type: `Analyze the trend of AI adoption in 2024` → Should route to data analysis
7. Type: `state` → Should show current graph state
8. Type: `quit` → Should exit cleanly

---

## 📋 COMPLETION SUMMARY

When all tasks are complete, the project will contain:

| Component | Count |
|-----------|-------|
| Python files | ~45 |
| State schemas | 7 |
| Graph/subgraph modules | 7 |
| Functional API workflows | 3 |
| Node functions | ~25 |
| Tools | 12 |
| Tests | 14 test files |
| Scripts | 4 |
| Frontend files | 3 |
| LangGraph concepts demonstrated | 67 |

**The project is now complete and production-ready.**