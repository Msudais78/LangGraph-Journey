"""
nodes/red_flag.py
=================
Node 5: Red Flag Detector (Dual-Layer)

Two-layer detection system:
  Layer 1 — Rule-based: Fast keyword scanning of all text
  Layer 2 — LLM-based:  Nuanced clinical analysis for ambiguous presentations
"""

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile
from medical_symptom_checker.models.assessment import SeverityAssessment
from medical_symptom_checker.nodes.utils import retry_llm_call, load_red_flags

logger = get_logger(__name__)


class RedFlagResult(BaseModel):
    """LLM structured output for red flag analysis."""
    red_flags_found: bool
    flags: list[str]
    reasoning: str


def _rule_based_scan(text: str, red_flags: dict) -> list[str]:
    """
    Layer 1: Scan text for red flag keywords.

    Returns list of flag descriptions for detected categories.
    """
    detected = []
    text_lower = text.lower()
    for category, keywords in red_flags.items():
        for keyword in keywords:
            if keyword in text_lower:
                flag = f"🚩 {category.upper().replace('_', ' ')}: '{keyword}'"
                detected.append(flag)
                logger.warning(f"Rule-based red flag detected: {flag}")
                break  # One per category
    return detected


def red_flag_detector_node(state: MedicalCheckerState) -> dict:
    """
    Node 5: Dual-layer red flag detection.

    Layer 1 runs first (fast, always). Layer 2 (LLM) adds nuanced detection.
    If either layer triggers, red_flag_triggered = True and severity = 'emergency'.

    Fallback: If LLM layer fails, rely on rule-based results alone.

    Args:
        state: Graph state with symptoms and messages.

    Returns:
        dict: red_flag_triggered boolean, updated severity_assessment, status message.
    """
    logger.info("=" * 55)
    logger.info("Node 5: red_flag_detector_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    messages = state.get("messages", [])
    patient = state.get("patient_profile", PatientProfile())

    # ── Build full text corpus for scanning ──
    symptom_text = " ".join([s.name for s in symptoms])
    message_text = " ".join([
        m.content if hasattr(m, "content") else m.get("content", "")
        for m in messages
    ])
    all_text = f"{symptom_text} {message_text}"

    # ── Layer 1: Rule-Based (always runs) ────────────────────────────
    red_flags_data = load_red_flags()
    rule_based_flags = _rule_based_scan(all_text, red_flags_data)
    logger.info(f"Layer 1 (rule-based): {len(rule_based_flags)} flag(s) detected")

    # ── Layer 2: LLM-Based (adds nuance) ─────────────────────────────
    llm_flags = []
    llm_flag_found = False

    try:
        llm = get_llm()
        patient_text = f"{patient.age}yo {patient.gender}" if patient.age else "Unknown"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an emergency triage specialist analyzing for CRITICAL red flags.
             Determine if ANY of the following emergency conditions may be present:

             🚨 IMMEDIATE RED FLAGS:
             - Signs of heart attack (chest pain+ jaw/arm pain + sweating)
             - Signs of stroke (F.A.S.T.: Face/Arm/Speech/Time)
             - Severe difficulty breathing or respiratory distress
             - Uncontrollable bleeding or blood from any orifice
             - Loss of consciousness, seizure, or altered mental status
             - Signs of anaphylaxis (throat swelling, severe allergic reaction)
             - Suspected meningitis (stiff neck + fever + severe headache + light sensitivity)
             - Suicidal ideation or self-harm intent
             - Suspected poisoning, overdose, or ingestion of harmful substances
             - Severe abdominal pain (especially with rigidity or rebound tenderness)
             - Signs of sepsis (high fever + rapid HR + confusion + history of infection)
             - Hypertensive crisis (BP > 180/120 with organ symptoms)

             Patient: {patient}
             Symptoms reported: {symptoms}
             Conversation summary: {conversation}

             Be conservative — when in doubt, flag as positive.
             Return structured JSON with: red_flags_found, flags (list), reasoning.
             """),
            ("human", "Analyze for emergency red flags.")
        ])

        chain = prompt | llm.with_structured_output(RedFlagResult)
        patient_text = f"{patient.age}yo {patient.gender}" if patient.age else "Unknown"
        symptoms_text = ", ".join([s.name for s in symptoms]) if symptoms else "None reported"
        conv_excerpt = all_text[:500]

        result: RedFlagResult = retry_llm_call(
            lambda: chain.invoke({
                "patient": patient_text,
                "symptoms": symptoms_text,
                "conversation": conv_excerpt,
            }),
            max_retries=2
        )

        llm_flags = result.flags if result.flags else []
        llm_flag_found = result.red_flags_found
        logger.info(
            f"Layer 2 (LLM): red_flags_found={llm_flag_found}, "
            f"{len(llm_flags)} flag(s) — {result.reasoning[:100]}"
        )

    except Exception as e:
        logger.warning(
            f"LLM red flag detection failed, relying on rule-based only: {e}"
        )

    # ── Merge results ──────────────────────────────────────────────
    all_flags = list(set(rule_based_flags + llm_flags))
    is_emergency = len(rule_based_flags) > 0 or llm_flag_found

    if is_emergency:
        logger.critical(
            f"🚨 RED FLAGS DETECTED! Total: {len(all_flags)} — "
            f"Routing to emergency guidance."
        )
    else:
        logger.info("✅ No red flags detected. Safe to continue normal assessment.")

    # Preserve any existing severity assessment
    existing_severity = state.get("severity_assessment", SeverityAssessment())
    updated_severity = SeverityAssessment(
        level="emergency" if is_emergency else existing_severity.level,
        red_flags_detected=all_flags,
        confidence_score=existing_severity.confidence_score,
        triage_score=existing_severity.triage_score,
        reasoning=existing_severity.reasoning,
        possible_conditions=existing_severity.possible_conditions,
    )

    status_msg = (
        f"🚨 RED FLAG ANALYSIS COMPLETE\n\n"
        f"{'⚠️ EMERGENCY FLAGS DETECTED:' if is_emergency else '✅ No immediate emergency flags found.'}\n"
    )
    if all_flags:
        status_msg += "\n".join([f"  {flag}" for flag in all_flags[:5]])

    if not is_emergency:
        status_msg += "\n\nProceeding to collect your medical history..."

    logger.info("red_flag_detector_node — COMPLETE ✅")
    return {
        "red_flag_triggered": is_emergency,
        "severity_assessment": updated_severity,
        "current_step": "red_flag_check",
        "messages": [{"role": "assistant", "content": status_msg}],
    }
