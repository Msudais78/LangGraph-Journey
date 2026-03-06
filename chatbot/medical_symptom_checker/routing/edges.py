"""
routing/edges.py
================
All conditional edge / routing functions for the medical checker graph.

Each function takes the current MedicalCheckerState and returns
the string name of the next node to execute.
"""

from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.assessment import SeverityAssessment
from medical_symptom_checker.models.patient import PatientProfile

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Route 1: Follow-Up Loop Control
# ══════════════════════════════════════════════

def should_continue_followup(state: MedicalCheckerState) -> str:
    """
    Decide whether to ask more follow-up questions or proceed.

    Routes:
        → "follow_up"       if needs_more_info AND loop < max_loops
        → "red_flag_check"  if info is sufficient OR loop limit reached
    """
    needs_more = state.get("needs_more_info", False)
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 5)

    if needs_more and loop_count < max_loops:
        logger.info(
            f"[Route] Follow-up needed — loop {loop_count + 1}/{max_loops} → follow_up"
        )
        return "follow_up"

    if loop_count >= max_loops:
        logger.warning(
            f"[Route] Follow-up loop limit reached ({max_loops}). "
            "Proceeding to red flag check."
        )
    else:
        logger.info("[Route] Symptom info sufficient → red_flag_check")

    return "red_flag_check"


# ══════════════════════════════════════════════
#  Route 2: Red Flag Fast-Path
# ══════════════════════════════════════════════

def route_after_red_flag(state: MedicalCheckerState) -> str:
    """
    Route based on red flag detection results.

    Routes:
        → "emergency_guidance"  if red_flag_triggered = True
        → "medical_history"     if no red flags detected
    """
    if state.get("red_flag_triggered", False):
        logger.critical(
            "[Route] 🚩 RED FLAG TRIGGERED → emergency_guidance (bypassing normal flow)"
        )
        return "emergency_guidance"

    logger.info("[Route] No red flags → medical_history")
    return "medical_history"


# ══════════════════════════════════════════════
#  Route 3: Core Severity Router ⭐
# ══════════════════════════════════════════════

def route_by_severity(state: MedicalCheckerState) -> str:
    """
    Core routing: direct to the appropriate care pathway based on severity level.

    Routes:
        → "emergency_guidance"    for 'emergency' or 'severe'
        → "doctor_recommendation" for 'moderate'
        → "home_remedy"           for 'mild'
    """
    severity = state.get("severity_assessment", SeverityAssessment())
    level = severity.level

    route_map = {
        "emergency": "emergency_guidance",
        "severe": "emergency_guidance",
        "moderate": "doctor_recommendation",
        "mild": "home_remedy",
    }

    destination = route_map.get(level, "home_remedy")
    logger.info(
        f"[Route] Severity: {level.upper()} "
        f"(confidence: {severity.confidence_score:.0%}) → {destination}"
    )
    return destination


# ══════════════════════════════════════════════
#  Route 4: Post Home-Remedy Router
# ══════════════════════════════════════════════

def route_after_home_remedy(state: MedicalCheckerState) -> str:
    """
    After home remedy node, branch based on original severity.

    Routes:
        → "lifestyle_prevention"   for MILD (direct mild path — full wellness coverage)
        → "drug_interaction_check" for MODERATE (came via specialist → skip lifestyle to drug check)
    """
    severity = state.get("severity_assessment", SeverityAssessment())

    if severity.level == "mild":
        logger.info("[Route] Mild pathway → lifestyle_prevention")
        return "lifestyle_prevention"
    else:
        logger.info(f"[Route] {severity.level.upper()} pathway → drug_interaction_check")
        return "drug_interaction_check"


# ══════════════════════════════════════════════
#  Route 5: Post Emergency Router
# ══════════════════════════════════════════════

def route_after_emergency(state: MedicalCheckerState) -> str:
    """
    After emergency guidance, check if we can look up nearby hospitals.

    Routes:
        → "hospital_finder"      if patient location is available
        → "mental_health_screen" if no location (skip useless search)
    """
    patient = state.get("patient_profile", PatientProfile())

    has_location = (
        patient is not None
        and hasattr(patient, "location")
        and bool(patient.location)
    )

    if has_location:
        logger.info(f"[Route] Location '{patient.location}' available → hospital_finder")
        return "hospital_finder"

    logger.info("[Route] No patient location → skipping hospital_finder → mental_health_screen")
    return "mental_health_screen"
