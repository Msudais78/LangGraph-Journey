"""
nodes/severity.py
=================
Node 7: Severity Classifier (Core Logic)

THE HEART OF THE SYSTEM — hybrid classification using:
1. Rule-based ESI (Emergency Severity Index) scoring
2. Gemini LLM clinical reasoning and condition matching
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile, MedicalHistory
from medical_symptom_checker.models.symptom import VitalSigns
from medical_symptom_checker.models.assessment import SeverityAssessment
from medical_symptom_checker.nodes.utils import retry_llm_call

logger = get_logger(__name__)


class ClassificationResult(BaseModel):
    """LLM structured output for severity classification."""
    level: Literal["mild", "moderate", "severe", "emergency"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    possible_conditions: list[dict]  # [{"name": "...", "probability": 0.8}]


def _compute_triage_score(
    patient: PatientProfile,
    symptoms: list,
    history: MedicalHistory,
    vitals: VitalSigns,
) -> tuple[int, list[str]]:
    """
    Rule-based ESI triage scoring (1=most urgent, 5=least).

    Returns:
        tuple: (triage_score, list of scoring notes for transparency)
    """
    score = 5  # Start at least urgent
    notes = []

    # ── Age risk ──
    if patient.is_high_risk_age:
        score -= 1
        notes.append(f"⚠️ High-risk age group ({patient.age}yo)")

    # ── Symptom intensity ──
    max_intensity = max((s.intensity for s in symptoms), default=0)
    if max_intensity >= 8:
        score -= 2
        notes.append(f"🔴 High intensity: {max_intensity}/10")
    elif max_intensity >= 5:
        score -= 1
        notes.append(f"🟡 Moderate intensity: {max_intensity}/10")

    # ── Acute onset ──
    if any(s.is_acute for s in symptoms):
        score -= 1
        notes.append("⚡ Acute/sudden onset detected")

    # ── Number of symptoms ──
    if len(symptoms) >= 4:
        score -= 1
        notes.append(f"📊 Multiple symptoms: {len(symptoms)}")

    # ── Chronic condition risk multiplier ──
    if history.has_high_risk_conditions:
        score -= 1
        notes.append("🏥 High-risk chronic conditions present")

    # ── Vital signs ──
    if vitals.has_high_fever:
        score -= 1
        notes.append(f"🌡️ High fever: {vitals.temperature_f}°F")
    if vitals.has_low_oxygen:
        score -= 2
        notes.append(f"😮‍💨 Low oxygen: {vitals.oxygen_saturation}%")
    if vitals.has_abnormal_hr:
        score -= 1
        notes.append(f"💓 Abnormal heart rate: {vitals.heart_rate_bpm} bpm")

    # ── Pregnancy ──
    if patient.pregnancy_status:
        score -= 1
        notes.append("🤰 Pregnancy — higher vigilance")

    # Clamp to valid ESI range
    score = max(1, min(5, score))
    return score, notes


def severity_classifier_node(state: MedicalCheckerState) -> dict:
    """
    Node 7: Classify clinical severity using hybrid scoring.

    Step 1: Rule-based ESI triage score
    Step 2: LLM clinical classification with full context
    Step 3: Merge and return final SeverityAssessment

    Fallback: Rule-based classification only (no LLM).

    Args:
        state: Full graph state with all collected data.

    Returns:
        dict: Updated severity_assessment with level, confidence, conditions.
    """
    logger.info("=" * 55)
    logger.info("Node 7: severity_classifier_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    patient = state.get("patient_profile", PatientProfile())
    history = state.get("medical_history", MedicalHistory())
    vitals = state.get("vital_signs", VitalSigns())
    existing = state.get("severity_assessment", SeverityAssessment())

    # ── Step 1: Rule-Based ESI Scoring ──────────────────────────────
    triage_score, scoring_notes = _compute_triage_score(patient, symptoms, history, vitals)
    logger.info(f"Rule-based ESI triage score: {triage_score}/5")
    for note in scoring_notes:
        logger.info(f"  Scoring note: {note}")

    # Rule-based severity estimate (for LLM context and fallback)
    if triage_score <= 1:
        rule_level = "emergency"
    elif triage_score == 2:
        rule_level = "severe"
    elif triage_score == 3:
        rule_level = "moderate"
    else:
        rule_level = "mild"

    # ── Step 2: LLM Classification ──────────────────────────────────
    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an expert clinical triage AI assistant (for educational purposes only).
             Classify the symptom severity level based on ALL provided information.

             SEVERITY CRITERIA:

             🟢 MILD (ESI 4-5):
             - Common cold, minor aches, mild headache, small cuts
             - Symptoms < 3 days, low intensity (1-3/10)
             - No fever or low-grade fever (< 100.4°F)
             - No impact on daily activities, no significant risk factors

             🟡 MODERATE (ESI 3):
             - Persistent fever (100.4-103°F), symptoms 3-7 days
             - Moderate pain (4-6/10), some functional impairment
             - Mild dehydration signs, worrying trend but not acute
             - Risk factors present (age > 65, chronic conditions)

             🔴 SEVERE (ESI 2):
             - High fever (> 103°F), severe pain (7-9/10)
             - Symptoms worsening rapidly over hours
             - Significant functional impairment
             - Signs of serious infection or organ involvement
             - Multiple high-risk factors combined

             🚨 EMERGENCY (ESI 1):
             - Any confirmed red flags
             - Life-threatening symptom combination
             - Altered consciousness or severe respiratory distress
             - 10/10 pain or "worst pain of my life"

             PATIENT DATA:
             - Profile: {patient}
             - Symptoms: {symptoms}
             - Medical History: {history}
             - Vital Signs: {vitals}
             - Rule-Based ESI Score: {triage_score}/5 (suggests: {rule_level})

             IMPORTANT: When uncertain, err toward HIGHER severity for patient safety.
             Provide top 3 possible conditions with probability estimates.
             """),
            ("human", "Classify severity and identify possible conditions.")
        ])

        chain = prompt | llm.with_structured_output(ClassificationResult)

        result: ClassificationResult = retry_llm_call(
            lambda: chain.invoke({
                "patient": patient.model_dump(),
                "symptoms": [s.model_dump() for s in symptoms],
                "history": history.model_dump(),
                "vitals": vitals.model_dump(),
                "triage_score": triage_score,
                "rule_level": rule_level,
            }),
            max_retries=2
        )

        logger.info(
            f"LLM classification: {result.level.upper()} "
            f"(confidence: {result.confidence_score:.0%})"
        )
        logger.info(f"Reasoning: {result.reasoning[:150]}")

        severity = SeverityAssessment(
            level=result.level,
            confidence_score=result.confidence_score,
            triage_score=triage_score,
            reasoning=result.reasoning,
            red_flags_detected=existing.red_flags_detected,
            possible_conditions=result.possible_conditions,
        )

    except Exception as e:
        logger.warning(f"LLM classification failed, using rule-based: {e}")

        # Rule-based fallback
        severity = SeverityAssessment(
            level=rule_level,
            confidence_score=0.6,  # Lower confidence without LLM
            triage_score=triage_score,
            reasoning=f"Rule-based classification (ESI score {triage_score}/5). "
                       f"Notes: {'; '.join(scoring_notes) or 'Standard assessment'}",
            red_flags_detected=existing.red_flags_detected,
            possible_conditions=[],
        )

    # ── Build output message ──────────────────────────────────────
    emoji = severity.severity_emoji
    conditions_text = "\n".join([f"  • {c}" for c in severity.top_conditions])

    output = (
        f"\n{'═' * 55}\n"
        f"{emoji}  SEVERITY ASSESSMENT: {severity.level.upper()}\n"
        f"{'═' * 55}\n\n"
        f"📊 Confidence: {severity.confidence_score:.0%}\n"
        f"🏥 ESI Triage Score: {severity.triage_score}/5\n\n"
        f"📋 Clinical Reasoning:\n{severity.reasoning}\n"
    )

    if severity.possible_conditions:
        output += f"\n🔍 Possible Conditions:\n{conditions_text}\n"

    if scoring_notes:
        output += f"\n📌 Risk Factors Identified:\n"
        output += "\n".join([f"  {note}" for note in scoring_notes])

    output += f"\n\n⚠️ Educational assessment only — consult a healthcare professional."

    logger.info(f"severity_classifier_node — COMPLETE ✅ ({severity.level.upper()})")
    return {
        "severity_assessment": severity,
        "current_step": "severity_classifier",
        "messages": [{"role": "assistant", "content": output}],
    }
