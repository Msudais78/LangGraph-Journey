"""
nodes/greeting.py
=================
Node 1: Greeting & Disclaimer

The first node in the graph. Displays the medical disclaimer and
initializes the session metadata. No LLM call needed — static content.
"""

from datetime import datetime

from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.nodes.utils import load_disclaimer

logger = get_logger(__name__)


def greeting_disclaimer_node(state: MedicalCheckerState) -> dict:
    """
    Node 1: Display medical disclaimer and greet the user.

    This is a deterministic node — no LLM call.
    Always succeeds (returns static content regardless of state).

    Args:
        state: Current graph state (mostly empty at this point)

    Returns:
        dict: Initial session fields + disclaimer message
    """
    logger.info("=" * 55)
    logger.info("Node 1: greeting_disclaimer_node — STARTING")
    logger.info("=" * 55)

    try:
        # Load disclaimer from file (with inline fallback)
        disclaimer_text = load_disclaimer()

        greeting_message = (
            f"{disclaimer_text}\n\n"
            "👋 Hello! I'm your educational health symptom assistant.\n\n"
            "To help assess your symptoms, I'll need to collect some information.\n"
            "Please tell me:\n"
            "  • Your name, age, and gender\n"
            "  • Your general location (city/state)\n"
            "  • The symptoms you're experiencing\n"
            "  • Any current medications or allergies\n\n"
            "Please share as much detail as you're comfortable with."
        )

        result = {
            "current_step": "greeting",
            "disclaimer_acknowledged": True,
            "timestamp": datetime.now().isoformat(),
            "loop_count": 0,
            "max_loops": 5,
            "messages": [{"role": "assistant", "content": greeting_message}],
        }

        logger.info("greeting_disclaimer_node — COMPLETE ✅")
        return result

    except Exception as e:
        # This node should never fail, but handle just in case
        logger.error(f"greeting_disclaimer_node encountered an error: {e}", exc_info=True)
        return {
            "current_step": "greeting",
            "disclaimer_acknowledged": True,
            "timestamp": datetime.now().isoformat(),
            "loop_count": 0,
            "max_loops": 5,
            "messages": [{
                "role": "assistant",
                "content": (
                    "⚠️ DISCLAIMER: Educational purposes only. Not medical advice. "
                    "Consult a healthcare professional. Call 911 in emergencies.\n\n"
                    "Please describe your symptoms."
                )
            }],
        }
