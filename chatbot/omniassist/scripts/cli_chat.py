"""Enhanced CLI chat interface for OmniAssist.

Usage: python scripts/cli_chat.py
"""

import sys
import os
import uuid

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
    table = Table(title="Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    commands = [
        ("quit", "Exit the chat"),
        ("new", "Start a new conversation thread"),
        ("history", "Show conversation history"),
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
    console.print(Panel.fit(
        "[bold green]🤖 OmniAssist[/bold green] — Powered by [cyan]Groq[/cyan]\n"
        "Type 'help' for commands. Type 'quit' to exit.",
        title="Welcome",
    ))

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    graph = get_main_graph(checkpointer=checkpointer, store=store)

    thread_id = str(uuid.uuid4())
    assistant = "default"
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Thread: {thread_id[:8]}... | Model: llama-3.3-70b-versatile[/dim]\n")

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

        if user_input.lower() == "help":
            print_help()
            continue

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            console.print(f"\n[green]New thread started: {thread_id[:8]}...[/green]\n")
            continue

        if user_input.lower() == "assistants":
            for name, cfg in ASSISTANT_CONFIGS.items():
                console.print(f"  [cyan]{name}[/cyan]: {cfg.get('persona', '?')} (model: {cfg.get('model_name', '?')})")
            continue

        if user_input.lower().startswith("switch "):
            new_assistant = user_input[7:].strip()
            if new_assistant in ASSISTANT_CONFIGS:
                assistant = new_assistant
                console.print(f"[green]Switched to: {assistant}[/green]")
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
                    role = "You" if isinstance(msg, HumanMessage) else "AI"
                    content = msg.content[:150] if hasattr(msg, "content") else str(msg)[:150]
                    console.print(f"  [{role}]: {content}")
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

        # Normal message
        try:
            with console.status("[italic dim]Thinking...[/italic dim]", spinner="dots"):
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
                agent = result.get("current_agent", "assistant")
                console.print(f"\n[dim]◆ {agent}[/dim]")
                console.print(f"[bold green]Assistant:[/bold green] ", end="")
                console.print(Markdown(last_ai_msg.content))
                console.print()
            else:
                console.print("[dim]No response generated.[/dim]\n")

        except Exception as e:
            error_str = str(e)
            if "interrupt" in error_str.lower():
                console.print(f"\n[yellow]⏸ Paused — operation requires approval.[/yellow]")
                console.print("[yellow]Use 'resume approved' or 'resume rejected'.[/yellow]\n")
            else:
                console.print(f"\n[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
