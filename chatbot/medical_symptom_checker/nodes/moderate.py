"""
nodes/moderate.py
=================
Nodes 9a & 9b: Moderate Pathway — Doctor Visit Recommender + Specialist Mapper

The MODERATE severity pathway.
9a: Recommends when/how to see a doctor.
9b: Maps symptoms to the appropriate medical specialist.
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile
from medical_symptom_checker.models.assessment import SeverityAssessment, Recommendation
from medical_symptom_checker.nodes.utils import retry_llm_call, load_specialist_map, format_symptoms_summary

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Node 9a: Doctor Visit Recommender
# ══════════════════════════════════════════════

def doctor_visit_recommender_node(state: MedicalCheckerState) -> dict:
    """
    Node 9a: Recommend appropriate doctor visit with timeline and preparation.

    Provides:
    - When to see a doctor (urgency window)
    - Type of appointment (urgent care vs primary care vs specialist)
    - What to prepare for the visit
    - Warning signs requiring immediate escalation to ER

    Fallback: Generic "see doctor within 2-3 days" message.

    Args:
        state: Graph state with symptoms, severity, and patient profile.

    Returns:
        dict: Doctor visit recommendation message.
    """
    logger.info("=" * 55)
    logger.info("Node 9a: doctor_visit_recommender_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    severity = state.get("severity_assessment", SeverityAssessment())
    patient = state.get("patient_profile", PatientProfile())

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a healthcare navigation assistant helping a patient with MODERATE symptoms.
             Provide a clear, structured doctor visit plan.

             Structure your response as:

             ⏰ RECOMMENDED TIMELINE:
             [Within 24 hours / 1-2 days / 2-3 days / within a week — with clear reasoning]

             🩺 TYPE OF VISIT:
             [Urgent care / Primary care / Telemedicine — which is most appropriate and why]

             📋 WHAT TO PREPARE:
             [Symptom diary checklist, documents to bring, insurance info]

             ❓ QUESTIONS TO ASK YOUR DOCTOR:
             [3-5 key questions relevant to their specific symptoms]

             ⚠️ GO TO ER IMMEDIATELY IF:
             [Specific red-flag warning signs that require escalation]

             Keep each section concise. Tailor advice to the patient's specific symptoms.
             Always remind: this is educational — follow professional medical advice.

             Patient: {patient}
             Symptoms: {symptoms}
             Severity: {severity}
             Possible conditions to investigate: {conditions}
             """),
            ("human", "Provide a doctor visit recommendation plan.")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "patient": f"{patient.age}yo {patient.gender}" if patient.age else "Unknown",
                "symptoms": format_symptoms_summary(symptoms),
                "severity": severity.level,
                "conditions": severity.top_conditions or ["To be determined by doctor"],
            }),
            max_retries=2
        )

        content = f"🩺 DOCTOR VISIT RECOMMENDATION\n\n{response.content}"
        logger.info("doctor_visit_recommender_node — COMPLETE ✅")

        return {
            "current_step": "doctor_recommendation",
            "messages": [{"role": "assistant", "content": content}],
        }

    except Exception as e:
        logger.error(f"doctor_visit_recommender_node failed: {e}", exc_info=True)
        return {
            "current_step": "doctor_recommendation",
            "messages": [{
                "role": "assistant",
                "content": (
                    "🩺 DOCTOR VISIT RECOMMENDATION\n\n"
                    "⏰ TIMELINE: See a doctor within 1-3 days.\n\n"
                    "🩺 TYPE: Start with your primary care physician or urgent care.\n\n"
                    "📋 PREPARE: Write down your symptoms, duration, severity, "
                    "and current medications before your appointment.\n\n"
                    "⚠️ GO TO ER IMMEDIATELY if symptoms worsen significantly, "
                    "you develop difficulty breathing, chest pain, or very high fever."
                )
            }],
        }


# ══════════════════════════════════════════════
#  Node 9b: Specialist Mapper
# ══════════════════════════════════════════════

def specialist_mapper_node(state: MedicalCheckerState) -> dict:
    """
    Node 9b: Map symptoms to the most appropriate medical specialist.

    Uses specialist mapping data + LLM reasoning to recommend
    the right specialist type for the patient's specific condition.

    Fallback: Keyword lookup from specialist_map.json.

    Args:
        state: Graph state with symptoms and assessment.

    Returns:
        dict: Updated recommendation with specialist_type + message.
    """
    logger.info("=" * 55)
    logger.info("Node 9b: specialist_mapper_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    severity = state.get("severity_assessment", SeverityAssessment())
    current_rec = state.get("recommendation", Recommendation())

    specialist_map = load_specialist_map()

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a medical specialist referral advisor.
             Based on the patient's symptoms and likely conditions, recommend the MOST APPROPRIATE specialist.

             Specialist mapping reference: {specialist_map}

             Provide:
             1. 🥼 PRIMARY SPECIALIST: [Type & why]
             2. 🔄 ALTERNATIVE: [If primary not available]
             3. 🏥 START WITH GP?: [Should they see a generalist first? Yes/No + brief reason]
             4. 📝 WHAT TO TELL THE SPECIALIST: [Key points for the referral]

             Be specific to their symptoms. Explain WHY this specialist is appropriate.
             Keep response under 200 words.
             """),
            ("human",
             "Symptoms: {symptoms}\n"
             "Possible conditions: {conditions}\n\n"
             "Which specialist should this patient see?")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "specialist_map": specialist_map,
                "symptoms": [{"name": s.name, "location": s.body_location} for s in symptoms],
                "conditions": severity.top_conditions or ["General illness"],
            }),
            max_retries=2
        )

        specialist_content = response.content
        logger.info(f"Specialist recommendation: {specialist_content[:100]}")

    except Exception as e:
        logger.warning(f"LLM specialist mapping failed, using keyword lookup: {e}")

        # Keyword fallback — find first match in specialist map
        specialist_type = "Primary Care Physician (GP)"
        for symptom in symptoms:
            symptom_keywords = [symptom.name, symptom.body_location]
            for keyword in symptom_keywords:
                for map_key, specialist in specialist_map.items():
                    if map_key in keyword.lower():
                        specialist_type = specialist
                        break

        specialist_content = (
            f"🥼 PRIMARY SPECIALIST: {specialist_type}\n\n"
            "Based on your reported symptoms. Your primary care physician can "
            "provide a formal referral if needed."
        )

    # Update recommendation with specialist type
    try:
        current_rec.specialist_type = specialist_content[:200]
    except Exception:
        pass

    content = f"👨‍⚕️ SPECIALIST RECOMMENDATION\n\n{specialist_content}"
    logger.info("specialist_mapper_node — COMPLETE ✅")

    return {
        "recommendation": current_rec,
        "current_step": "specialist_mapper",
        "messages": [{"role": "assistant", "content": content}],
    }
