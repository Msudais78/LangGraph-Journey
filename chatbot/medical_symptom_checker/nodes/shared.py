"""
nodes/shared.py
===============
Shared Pathway Nodes (Nodes 11-14) — all pathways converge here.

Contains:
- drug_interaction_node        (Node 11): Medication safety check
- mental_health_screening_node (Node 12): Basic mental health screen
- summary_report_node          (Node 13): Comprehensive report generator
- feedback_followup_node       (Node 14): Follow-up schedule + feedback
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile, MedicalHistory
from medical_symptom_checker.models.assessment import SeverityAssessment, Recommendation
from medical_symptom_checker.models.mental_health import MentalHealthScreen
from medical_symptom_checker.nodes.utils import retry_llm_call

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Node 11: Drug Interaction Checker
# ══════════════════════════════════════════════

def drug_interaction_node(state: MedicalCheckerState) -> dict:
    """
    Node 11: Check for potential drug interactions.

    Cross-checks:
    - Patient's current medications vs. OTC suggestions
    - Drug allergies
    - Condition-specific contraindications

    Fallback: "Consult a pharmacist" message.

    Args:
        state: Graph state with medical history and recommendation.

    Returns:
        dict: Drug interaction warning message.
    """
    logger.info("=" * 55)
    logger.info("Node 11: drug_interaction_node — STARTING")
    logger.info("=" * 55)

    history = state.get("medical_history", MedicalHistory())
    recommendation = state.get("recommendation", Recommendation())

    current_meds = history.current_medications
    suggested_meds = recommendation.otc_medications
    allergies = history.allergies
    conditions = history.chronic_conditions

    # Skip lengthy LLM call if no medications are involved
    if not current_meds and not suggested_meds and not allergies:
        logger.info("No medications or allergies to check — skipping LLM call")
        return {
            "current_step": "drug_interaction_check",
            "messages": [{
                "role": "assistant",
                "content": (
                    "💊 MEDICATION SAFETY CHECK\n\n"
                    "✅ No current medications or known allergies on record.\n\n"
                    "If you take any medications (including OTC or supplements), "
                    "always check with your pharmacist before adding new ones.\n\n"
                    "⚠️ This educational tool cannot replace professional pharmacist advice."
                )
            }],
        }

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a clinical pharmacist AI assistant checking medication safety.
             ⚠️ This is for EDUCATIONAL PURPOSES ONLY. Not a substitute for professional advice.

             Check for:
             1. Drug-drug interactions between current and suggested medications
             2. Drug-allergy conflicts
             3. Condition-specific contraindications
                (e.g., NSAIDs with kidney disease, blood thinners with aspirin)
             4. Safe alternatives if conflicts found

             CURRENT MEDICATIONS: {current_meds}
             SUGGESTED OTC MEDICATIONS: {suggested_meds}
             KNOWN ALLERGIES: {allergies}
             CHRONIC CONDITIONS: {conditions}

             Format response:

             ✅ SAFE TO TAKE:
             [List of suggested medications that appear safe]

             ⚠️ POTENTIAL CONCERNS:
             [Interactions found with severity: minor/moderate/severe/contraindicated]

             🔄 SAFER ALTERNATIVES:
             [If any concerns found]

             📋 RECOMMENDATIONS:
             [Always recommend consulting pharmacist or doctor before any medication change]

             If no medications to check, confirm safety with general guidance.
             """),
            ("human", "Check drug interactions and safety.")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "current_meds": current_meds or ["None reported"],
                "suggested_meds": suggested_meds or ["None suggested"],
                "allergies": allergies or ["None reported"],
                "conditions": conditions or ["None reported"],
            }),
            max_retries=2
        )

        content = f"💊 MEDICATION SAFETY CHECK\n\n{response.content}"
        logger.info("drug_interaction_node — COMPLETE ✅")

        return {
            "current_step": "drug_interaction_check",
            "messages": [{"role": "assistant", "content": content}],
        }

    except Exception as e:
        logger.error(f"drug_interaction_node failed: {e}", exc_info=True)
        return {
            "current_step": "drug_interaction_check",
            "messages": [{
                "role": "assistant",
                "content": (
                    "💊 MEDICATION SAFETY CHECK\n\n"
                    "⚠️ Unable to perform automated drug interaction check.\n\n"
                    "📋 IMPORTANT: Before taking any new medication:\n"
                    "  • Consult your pharmacist (free at most pharmacies)\n"
                    "  • Inform them of ALL current medications and supplements\n"
                    "  • Mention all known allergies\n\n"
                    "Your pharmacist is your medication safety expert — always ask them first."
                )
            }],
        }


# ══════════════════════════════════════════════
#  Node 12: Mental Health Screening
# ══════════════════════════════════════════════

def mental_health_screening_node(state: MedicalCheckerState) -> dict:
    """
    Node 12: Basic mental health screening checkpoint.

    Screens for stress, anxiety, depression indicators.
    CRITICAL: Any mention of suicidal ideation → immediate crisis resources.

    Fallback: Static crisis resources + gentle referral.

    Args:
        state: Graph state with symptoms and messages.

    Returns:
        dict: Mental health screening result + supportive message.
    """
    logger.info("=" * 55)
    logger.info("Node 12: mental_health_screening_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    messages = state.get("messages", [])

    # ── Crisis keyword pre-check ──────────────────────────────────
    crisis_keywords = [
        "suicidal", "kill myself", "end my life", "don't want to live",
        "self harm", "harming myself", "overdose on purpose", "want to die"
    ]
    all_text = " ".join([
        m.content if hasattr(m, "content") else m.get("content", "")
        for m in messages
    ]).lower()

    crisis_detected = any(kw in all_text for kw in crisis_keywords)

    if crisis_detected:
        logger.critical("🆘 CRISIS DETECTED — Providing immediate crisis resources")
        crisis_msg = (
            "🆘 MENTAL HEALTH CRISIS RESOURCES\n\n"
            "I'm concerned about your wellbeing. Please reach out for help right now:\n\n"
            "📞 988 Suicide & Crisis Lifeline:\n"
            "   Call or Text: 988 (US) — Available 24/7\n\n"
            "📱 Crisis Text Line:\n"
            "   Text HOME to 741741 — Available 24/7\n\n"
            "🚨 If in immediate danger: Call 911\n\n"
            "💙 You matter. Help is available. You don't have to face this alone.\n\n"
            "Please reach out to one of these resources before continuing."
        )
        screen = MentalHealthScreen(
            anxiety_indicators=True,
            depression_indicators=True,
            recommendation=crisis_msg,
        )
        return {
            "mental_health_screen": screen,
            "current_step": "mental_health_screen",
            "messages": [{"role": "assistant", "content": crisis_msg}],
        }

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a compassionate mental health screening assistant.
             Perform a BRIEF, non-intrusive mental health check based on the symptoms described.

             Note: Many physical symptoms (headache, fatigue, stomach issues, insomnia)
             can have mental health components or be worsened by stress/anxiety.

             Screen gently for:
             - Stress indicators (work, family, financial pressures)
             - Anxiety signs (excessive worry, restlessness, sleep difficulties)
             - Depression indicators (low mood, fatigue, loss of interest)
             - Sleep quality issues
             - Appetite changes

             IMPORTANT GUIDELINES:
             - Be warm, non-judgmental, and supportive
             - Do NOT diagnose
             - If ANY suicidal ideation detected → provide 988 immediately
             - Normalize seeking help
             - Keep response brief (under 150 words)

             Physical symptoms reported: {symptoms}
             """),
            ("human", "Perform a gentle mental health wellness check.")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "symptoms": [s.name for s in symptoms],
            }),
            max_retries=2
        )

        screening_text = response.content

        # Always include crisis resources at the bottom
        full_content = (
            f"🧠 MENTAL HEALTH & WELLNESS CHECK\n\n"
            f"{screening_text}\n\n"
            f"{'─' * 40}\n"
            f"📌 REMINDER: Mental health support is available:\n"
            f"   • Call/Text 988 — Suicide & Crisis Lifeline\n"
            f"   • Crisis Text: Text HOME to 741741"
        )

        screen = MentalHealthScreen(recommendation=screening_text)
        logger.info("mental_health_screening_node — COMPLETE ✅")

        return {
            "mental_health_screen": screen,
            "current_step": "mental_health_screen",
            "messages": [{"role": "assistant", "content": full_content}],
        }

    except Exception as e:
        logger.error(f"mental_health_screening_node failed: {e}", exc_info=True)
        static_content = (
            "🧠 MENTAL HEALTH & WELLNESS CHECK\n\n"
            "Your mental wellbeing is just as important as your physical health.\n\n"
            "Physical symptoms can often be connected to or worsened by stress, "
            "anxiety, or poor sleep. If you've been feeling overwhelmed, it's okay "
            "to seek support.\n\n"
            "📌 Mental health resources:\n"
            "   • Call/Text 988 — Suicide & Crisis Lifeline (24/7)\n"
            "   • Crisis Text: Text HOME to 741741 (24/7)\n"
            "   • Consider speaking with your doctor about how you're feeling emotionally too."
        )
        return {
            "mental_health_screen": MentalHealthScreen(recommendation=static_content),
            "current_step": "mental_health_screen",
            "messages": [{"role": "assistant", "content": static_content}],
        }


# ══════════════════════════════════════════════
#  Node 13: Summary Report Generator
# ══════════════════════════════════════════════

def summary_report_node(state: MedicalCheckerState) -> dict:
    """
    Node 13: Generate a comprehensive, human-readable health summary report.

    No LLM call — pure string formatting from state data.
    Always succeeds (static template).

    Args:
        state: Full graph state with all collected data.

    Returns:
        dict: final_report string + formatted report message.
    """
    logger.info("=" * 55)
    logger.info("Node 13: summary_report_node — STARTING")
    logger.info("=" * 55)

    patient = state.get("patient_profile", PatientProfile())
    symptoms = state.get("symptoms", [])
    history = state.get("medical_history", MedicalHistory())
    severity = state.get("severity_assessment", SeverityAssessment())
    recommendation = state.get("recommendation", Recommendation())
    mental = state.get("mental_health_screen", MentalHealthScreen())
    session_id = state.get("session_id", "N/A")
    timestamp = state.get("timestamp", "N/A")

    # ── Build sections ─────────────────────────────────────────
    symptoms_section = "\n".join([
        f"  • {s.name} (Intensity: {s.intensity}/10"
        + (f", Duration: {s.duration}" if s.duration else "")
        + (f", Location: {s.body_location}" if s.body_location else "")
        + ")"
        for s in symptoms
    ]) or "  None recorded"

    conditions_section = "\n".join([
        f"     • {c.get('name', 'Unknown')} ({c.get('probability', 'N/A')})"
        for c in severity.possible_conditions
    ]) or "     • To be determined by a healthcare professional"

    red_flags_section = (
        ", ".join(severity.red_flags_detected) if severity.red_flags_detected else "None detected"
    )

    meds_section = (
        ", ".join(history.current_medications) if history.current_medications else "None reported"
    )

    allergies_section = (
        ", ".join(history.allergies) if history.allergies else "None reported"
    )

    conditions_chronic_section = (
        ", ".join(history.chronic_conditions) if history.chronic_conditions else "None reported"
    )

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           📋  HEALTH ASSESSMENT SUMMARY REPORT                  ║
╠══════════════════════════════════════════════════════════════════╣
║    ⚠️  EDUCATIONAL PURPOSE ONLY — NOT MEDICAL ADVICE            ║
╚══════════════════════════════════════════════════════════════════╝

📅 Date:       {timestamp[:19].replace('T', ' ') if 'T' in timestamp else timestamp}
🆔 Session:    {session_id}

{"═" * 60}
👤 PATIENT PROFILE
{"═" * 60}
  Name:     {patient.name or "Not provided"}
  Age:      {patient.age or "Not provided"} | Gender: {patient.gender or "Not provided"}
  Location: {patient.location or "Not provided"}

{"═" * 60}
🤒 SYMPTOMS REPORTED
{"═" * 60}
{symptoms_section}

{"═" * 60}
📊 SEVERITY ASSESSMENT
{"═" * 60}
  {severity.severity_emoji}  Level:      {severity.level.upper()}
     Confidence:  {severity.confidence_score:.0%}
     ESI Score:   {severity.triage_score}/5 (1=most urgent)
     Red Flags:   {red_flags_section}

  📋 Clinical Reasoning:
     {severity.reasoning[:300] if severity.reasoning else "See individual assessment steps above"}

  🔍 Possible Conditions:
{conditions_section}

{"═" * 60}
💡 RECOMMENDATIONS
{"═" * 60}
  Primary Action:   {recommendation.primary_action or "Consult a healthcare professional"}
  Urgency:          {recommendation.urgency_timeframe or "As appropriate for your symptoms"}
  Specialist:       {recommendation.specialist_type[:100] if recommendation.specialist_type else "As directed by your doctor"}

{"═" * 60}
📋 MEDICAL BACKGROUND
{"═" * 60}
  Chronic Conditions: {conditions_chronic_section}
  Medications:        {meds_section}
  Allergies:          {allergies_section}

{"═" * 60}
🧠 MENTAL HEALTH & WELLNESS
{"═" * 60}
  Status: {"⚠️ Potential concerns noted — please seek support" if mental.needs_referral else "✅ No significant concerns identified"}
  Crisis Line: 988 (US) | Text HOME to 741741

{"═" * 60}
⚠️  IMPORTANT DISCLAIMER
{"═" * 60}
  This is an AI-generated EDUCATIONAL report.
  It does NOT constitute medical advice, diagnosis, or treatment.
  Always consult a licensed healthcare professional for medical decisions.
  In emergencies: Call 911 immediately.
{"═" * 60}
"""

    logger.info(f"Summary report generated ({len(report)} chars)")
    logger.info("summary_report_node — COMPLETE ✅")

    return {
        "final_report": report,
        "current_step": "summary_report",
        "messages": [{"role": "assistant", "content": report}],
    }


# ══════════════════════════════════════════════
#  Node 14: Feedback & Follow-Up Scheduler
# ══════════════════════════════════════════════

def feedback_followup_node(state: MedicalCheckerState) -> dict:
    """
    Node 14: Provide follow-up schedule and collect feedback.

    Deterministic node — no LLM call. Always succeeds.

    Args:
        state: Graph state with severity assessment.

    Returns:
        dict: Follow-up schedule + closing message.
    """
    logger.info("=" * 55)
    logger.info("Node 14: feedback_followup_node — STARTING")
    logger.info("=" * 55)

    severity = state.get("severity_assessment", SeverityAssessment())

    FOLLOW_UP_SCHEDULE = {
        "mild": (
            "Self-monitor for 3-5 days.\n"
            "  • Check your temperature daily if feverish\n"
            "  • Note any new or worsening symptoms\n"
            "  • If no improvement in 5 days → see a doctor\n"
            "  • If symptoms worsen significantly → see a doctor sooner"
        ),
        "moderate": (
            "Follow up with a doctor within 1-3 days.\n"
            "  • Monitor symptoms daily\n"
            "  • Keep a symptom diary to share with your doctor\n"
            "  • Note any changes (better or worse)\n"
            "  • If symptoms significantly worsen → go to urgent care or ER"
        ),
        "severe": (
            "Follow up after emergency/specialist visit ASAP.\n"
            "  • Do not delay seeking professional care\n"
            "  • Have someone accompany you if feeling unwell\n"
            "  • Keep all follow-up appointments"
        ),
        "emergency": (
            "Seek emergency care NOW — do not delay follow-up.\n"
            "  • Call 911 or go to the nearest ER immediately\n"
            "  • Follow medical staff's instructions completely\n"
            "  • Reschedule non-essential activities until assessed"
        ),
    }

    schedule = FOLLOW_UP_SCHEDULE.get(severity.level, FOLLOW_UP_SCHEDULE["mild"])

    closing_message = (
        f"📅 FOLLOW-UP SCHEDULE\n\n"
        f"{schedule}\n\n"
        f"{'─' * 55}\n\n"
        f"💬 Thank you for using the Medical Symptom Checker.\n\n"
        f"📌 KEY REMINDERS:\n"
        f"  • This was an EDUCATIONAL assessment only\n"
        f"  • Always consult a licensed healthcare professional\n"
        f"  • Save or print your Summary Report above for reference\n"
        f"  • In emergencies: Call 911 immediately\n"
        f"  • Mental health: Call/Text 988 anytime\n\n"
        f"{'─' * 55}\n"
        f"💚 Take care and stay well!\n"
        f"{'─' * 55}"
    )

    logger.info("feedback_followup_node — COMPLETE ✅")

    return {
        "current_step": "feedback_followup",
        "messages": [{"role": "assistant", "content": closing_message}],
    }
