"""
main.py
=======
Interactive CLI Entry Point for the Medical Symptom Checker.

Provides:
- Multi-turn conversation interface
- Streaming output (real-time responses)
- Demo mode with pre-built scenarios
- Graceful error handling
- Clean session management
"""

import sys
import time
import argparse
from datetime import datetime

from medical_symptom_checker.core.config import settings, ensure_dirs
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import create_initial_state
from medical_symptom_checker.graph import build_medical_checker_graph

logger = get_logger(__name__)

# ══════════════════════════════════════════════
#  CLI Color / Style Constants
# ══════════════════════════════════════════════
class Colors:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"


def cprint(text: str, color: str = Colors.RESET, bold: bool = False) -> None:
    """Print colored text to console."""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.RESET}")


# ══════════════════════════════════════════════
#  Demo Scenarios
# ══════════════════════════════════════════════
DEMO_SCENARIOS = {
    "1": {
        "name": "🤒 Mild Cold (Standard Path)",
        "messages": [
            "My name is Alex, I'm 28 years old, male, from Chicago.",
            "I have a runny nose, mild sore throat, and feel a bit tired. "
            "Symptoms started 2 days ago. No medications. No allergies.",
        ],
    },
    "2": {
        "name": "🔴 Emergency: Chest Pain (Fast-Path)",
        "messages": [
            "John, 58, male, Houston Texas.",
            "I have severe crushing chest pain radiating to my left arm. "
            "Sudden onset 20 minutes ago. I'm also sweating and feeling dizzy. "
            "I take lisinopril for hypertension. Intensity 9/10.",
        ],
    },
    "3": {
        "name": "🟡 Moderate: Persistent Fever",
        "messages": [
            "Maria, 45, female, New York.",
            "I've had a fever of 102°F for 4 days along with body aches, "
            "fatigue, and mild headache. I have type 2 diabetes and take metformin. "
            "No allergies. Not improving with rest and OTC medications.",
        ],
    },
    "4": {
        "name": "🧠 Mental Health + Physical Symptoms",
        "messages": [
            "Sam, 32, non-binary, Seattle.",
            "I've been having severe headaches every day for a week, "
            "difficulty sleeping, and I feel really anxious and overwhelmed with work. "
            "I'm also not eating much. Stress level is very high. "
            "Taking no medications.",
        ],
    },
    "5": {
        "name": "👵 High-Risk: Elderly Multiple Conditions",
        "messages": [
            "My name is Dorothy, I'm 72 years old, female, from Phoenix Arizona.",
            "I have chest tightness, shortness of breath when walking, "
            "and ankle swelling for the past 3 days. "
            "I have heart disease, diabetes, and kidney disease. "
            "I take warfarin, metformin, and furosemide. "
            "No known food allergies but allergic to penicillin. "
            "Pain level 6/10.",
        ],
    },
}


# ══════════════════════════════════════════════
#  Output Streaming Simulation
# ══════════════════════════════════════════════
def stream_text(text: str, delay: float = 0.008) -> None:
    """Print text with streaming effect (character by character)."""
    if settings.enable_streaming:
        for char in text:
            print(char, end="", flush=True)
            if char in (".", "!", "?", "\n"):
                time.sleep(delay * 5)
            else:
                time.sleep(delay)
        print()
    else:
        print(text)


def print_separator(char: str = "─", width: int = 60, color: str = Colors.DIM) -> None:
    """Print a styled separator line."""
    cprint(char * width, color=color)


# ══════════════════════════════════════════════
#  Graph Runner
# ══════════════════════════════════════════════
def run_assessment(user_messages: list[str], session_id: str = None) -> dict:
    """
    Run the full medical assessment graph with provided messages.

    Args:
        user_messages: List of user message strings (multi-turn conversation)
        session_id: Optional session identifier

    Returns:
        dict: Final graph state after completion
    """
    import uuid

    session_id = session_id or str(uuid.uuid4())
    graph = build_medical_checker_graph()

    # Build initial state with user messages
    initial_messages = [
        {"role": "user", "content": msg}
        for msg in user_messages
    ]

    initial_state = create_initial_state(
        messages=initial_messages,
        session_id=session_id,
    )

    final_state = {}

    cprint("\n🔄 Running Medical Assessment...", Colors.CYAN, bold=True)
    print_separator("─")

    try:
        # Stream through graph events
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_state in event.items():
                # Show which node is running
                cprint(f"\n▶ [{node_name}]", Colors.MAGENTA, bold=True)

                # Display assistant messages
                messages = node_state.get("messages", [])
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        print_separator()
                        cprint(msg.get("content", ""), Colors.WHITE)
                    elif hasattr(msg, "content") and hasattr(msg, "type"):
                        if msg.type == "ai":
                            print_separator()
                            cprint(msg.content, Colors.WHITE)

                # Track final state
                final_state.update(node_state)

    except KeyboardInterrupt:
        cprint("\n⚠️  Assessment interrupted by user.", Colors.YELLOW)
    except Exception as e:
        logger.error(f"Graph execution error: {e}", exc_info=True)
        cprint(f"\n⚠️  An error occurred: {str(e)[:200]}", Colors.RED)
        cprint("Please try again or consult a healthcare professional directly.", Colors.YELLOW)

    return final_state


# ══════════════════════════════════════════════
#  Interactive Mode
# ══════════════════════════════════════════════
def run_interactive() -> None:
    """
    Run the full interactive multi-turn conversation CLI.

    Collects two pieces of user input:
    1. Personal information (name, age, gender, location)
    2. Symptoms description (detailed)

    Then runs the graph and shows results.
    """
    print_separator("═", 65)
    cprint("🏥  MEDICAL SYMPTOM CHECKER (Educational Tool)", Colors.CYAN, bold=True)
    cprint(f"   Powered by Gemini {settings.llm_model} + LangGraph", Colors.DIM)
    print_separator("═", 65)

    cprint(
        "\n💡 TIP: Be as descriptive as possible for a better assessment.",
        Colors.YELLOW
    )
    cprint(
        "   Include: symptom name, severity (1-10), duration, onset, location on body.",
        Colors.DIM
    )
    print()

    user_messages = []

    # ── Step 1: Personal Information ─────────────────────────
    print_separator("─", 60)
    cprint("📝 STEP 1: Personal Information", Colors.GREEN, bold=True)
    print_separator("─", 60)
    cprint("Please share your name, age, gender, and location:", Colors.WHITE)
    cprint("(Example: 'John, 35 years old, male, from Dallas Texas')", Colors.DIM)
    print()

    try:
        personal_info = input(f"{Colors.CYAN}You → {Colors.RESET}").strip()
        if not personal_info:
            personal_info = "Anonymous patient"
        user_messages.append(personal_info)

        # ── Step 2: Symptoms ──────────────────────────────────
        print()
        print_separator("─", 60)
        cprint("🤒 STEP 2: Symptom Description", Colors.GREEN, bold=True)
        print_separator("─", 60)
        cprint("Describe your symptoms in detail:", Colors.WHITE)
        cprint(
            "(Include: what you feel, how long, how severe 1-10, "
            "medications, allergies, medical conditions)",
            Colors.DIM
        )
        print()

        symptoms_desc = input(f"{Colors.CYAN}You → {Colors.RESET}").strip()
        if not symptoms_desc:
            cprint("⚠️  No symptoms provided. Exiting.", Colors.YELLOW)
            return
        user_messages.append(symptoms_desc)

    except (KeyboardInterrupt, EOFError):
        cprint("\n👋 Goodbye! Stay healthy.", Colors.GREEN)
        return

    # ── Run Graph ─────────────────────────────────────────────
    print()
    final_state = run_assessment(user_messages)

    # ── Session Summary ───────────────────────────────────────
    print()
    print_separator("═", 65)
    cprint("✅  Assessment Complete", Colors.GREEN, bold=True)
    cprint(f"   Session: {final_state.get('session_id', 'N/A')[:8]}...", Colors.DIM)
    severity = final_state.get("severity_assessment")
    if severity:
        cprint(
            f"   Severity: {severity.severity_emoji} {severity.level.upper()}",
            Colors.YELLOW if severity.level == "mild" else Colors.RED
        )
    print_separator("═", 65)

    cprint(
        "\n⚠️  REMINDER: This was an educational assessment only.",
        Colors.YELLOW
    )
    cprint(
        "   Always consult a licensed healthcare professional for medical advice.",
        Colors.YELLOW
    )
    print()


# ══════════════════════════════════════════════
#  Demo Mode
# ══════════════════════════════════════════════
def run_demo(scenario_key: str = None) -> None:
    """
    Run a pre-built demonstration scenario.

    Args:
        scenario_key: Key from DEMO_SCENARIOS (e.g., "1", "2").
                      If None, prompts the user to choose.
    """
    print_separator("═", 65)
    cprint("🎭  DEMO MODE", Colors.MAGENTA, bold=True)
    cprint("   Pre-built medical scenarios for demonstration", Colors.DIM)
    print_separator("═", 65)

    if scenario_key not in DEMO_SCENARIOS:
        cprint("\nAvailable scenarios:", Colors.CYAN, bold=True)
        for key, scenario in DEMO_SCENARIOS.items():
            cprint(f"  [{key}] {scenario['name']}", Colors.WHITE)
        print()

        try:
            choice = input(f"{Colors.CYAN}Select scenario (1-{len(DEMO_SCENARIOS)}): {Colors.RESET}").strip()
            scenario_key = choice
        except (KeyboardInterrupt, EOFError):
            cprint("\n👋 Demo cancelled.", Colors.YELLOW)
            return

    scenario = DEMO_SCENARIOS.get(scenario_key)
    if not scenario:
        cprint(f"⚠️  Invalid scenario: {scenario_key}", Colors.RED)
        return

    cprint(f"\n🎬 Running: {scenario['name']}", Colors.CYAN, bold=True)
    cprint("─" * 60, Colors.DIM)

    # Show demo inputs
    cprint("\n📝 User Input:", Colors.GREEN)
    for i, msg in enumerate(scenario["messages"], 1):
        cprint(f"  [{i}] {msg}", Colors.DIM)
    print()

    # Run assessment
    run_assessment(scenario["messages"])


# ══════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════
def main() -> None:
    """Parse arguments and launch appropriate mode."""
    parser = argparse.ArgumentParser(
        prog="medical_symptom_checker",
        description="🏥 Medical Symptom Checker — Educational AI Tool (LangGraph + Gemini)",
    )
    parser.add_argument(
        "--demo",
        nargs="?",
        const="",
        metavar="SCENARIO",
        help="Run in demo mode. Optionally specify scenario number (1-5)."
    )
    parser.add_argument(
        "--version", action="version", version="Medical Symptom Checker v1.0.0"
    )

    args = parser.parse_args()

    # ── Validate Environment ─────────────────────────
    ensure_dirs()

    if not settings.google_api_key:
        cprint("\n❌ GOOGLE_API_KEY not set!", Colors.RED, bold=True)
        cprint("   1. Copy .env.example to e:\\AgenticAI\\.env", Colors.YELLOW)
        cprint("   2. Add your Gemini API key: GOOGLE_API_KEY=your-key", Colors.YELLOW)
        cprint("   3. Run again\n", Colors.YELLOW)
        sys.exit(1)

    logger.info(f"Starting Medical Symptom Checker — Model: {settings.llm_model}")

    # ── Launch Mode ──────────────────────────────────
    if args.demo is not None:
        run_demo(args.demo if args.demo else None)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
