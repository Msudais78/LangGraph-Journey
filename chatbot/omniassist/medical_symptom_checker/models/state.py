"""
models/state.py
===============
MedicalCheckerState TypedDict — the master state object for the entire graph.
All nodes read from and write to this shared type.
"""

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

from medical_symptom_checker.models.patient import PatientProfile, MedicalHistory
from medical_symptom_checker.models.symptom import Symptom, VitalSigns
from medical_symptom_checker.models.assessment import SeverityAssessment, Recommendation
from medical_symptom_checker.models.mental_health import MentalHealthScreen


class MedicalCheckerState(TypedDict, total=False):
    """
    Master state object that flows through the entire LangGraph.

    Every node:
    1. Reads needed fields from this state
    2. Returns a dict with ONLY the fields it's updating
    3. LangGraph merges updates back into this state

    The `Annotated[list, add_messages]` on `messages` means new messages
    are APPENDED rather than replaced (LangGraph message accumulation).
    """

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    current_step: str

    # ── Patient Info ──────────────────────────────────────────────────────────
    patient_profile: PatientProfile
    medical_history: MedicalHistory
    vital_signs: VitalSigns

    # ── Symptom Collection ────────────────────────────────────────────────────
    symptoms: list[Symptom]
    follow_up_questions: list[str]
    follow_up_answers: list[str]
    symptom_collection_complete: bool

    # ── Clinical Analysis ─────────────────────────────────────────────────────
    severity_assessment: SeverityAssessment
    recommendation: Recommendation
    mental_health_screen: MentalHealthScreen

    # ── Control Flow ──────────────────────────────────────────────────────────
    red_flag_triggered: bool
    needs_more_info: bool
    loop_count: int        # Tracks follow-up iterations
    max_loops: int         # Hard limit to prevent infinite loops

    # ── Output & Session ──────────────────────────────────────────────────────
    final_report: str
    disclaimer_acknowledged: bool
    feedback: str
    session_id: str
    timestamp: str


def create_initial_state(**overrides) -> dict:
    """
    Factory function: creates a properly initialized state dict with safe defaults.
    Use this instead of building the dict manually to avoid missing-field errors.

    Args:
        **overrides: Any state fields to set explicitly (e.g., messages=[...])

    Returns:
        dict: Complete initial state for the medical checker graph.

    Example:
        state = create_initial_state(
            messages=[{"role": "user", "content": "I have a headache"}],
            session_id="test-001"
        )
    """
    from datetime import datetime
    import uuid

    defaults: dict = {
        "messages": [],
        "current_step": "",
        "patient_profile": PatientProfile(),
        "medical_history": MedicalHistory(),
        "vital_signs": VitalSigns(),
        "symptoms": [],
        "follow_up_questions": [],
        "follow_up_answers": [],
        "symptom_collection_complete": False,
        "severity_assessment": SeverityAssessment(),
        "recommendation": Recommendation(),
        "mental_health_screen": MentalHealthScreen(),
        "red_flag_triggered": False,
        "needs_more_info": False,
        "loop_count": 0,
        "max_loops": 5,
        "final_report": "",
        "disclaimer_acknowledged": False,
        "feedback": "",
        "session_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
    }

    defaults.update(overrides)
    return defaults
