"""
nodes/mild.py
=============
Nodes 10a & 10b: Mild Pathway — Home Remedy + Lifestyle & Prevention

The MILD severity pathway.
10a: Evidence-based home remedies with allergy/medication safety checks.
10b: Lifestyle modifications and prevention tips for recovery and recurrence.
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile, MedicalHistory
from medical_symptom_checker.models.assessment import Recommendation
from medical_symptom_checker.nodes.utils import retry_llm_call, format_symptoms_summary

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Node 10a: Home Remedy Suggestion
# ══════════════════════════════════════════════

def home_remedy_node(state: MedicalCheckerState) -> dict:
    """
    Node 10a: Suggest evidence-based home care for mild symptoms.

    CRITICAL SAFETY: Always cross-checks against:
    - Patient allergies
    - Current medications (avoid interactions)
    - Chronic conditions (e.g., NOT recommending NSAIDs to someone with kidney disease)
    - Age-appropriate dosing
    - Pregnancy safety

    Fallback: Generic safe rest/hydration advice.

    Args:
        state: Graph state with all patient data.

    Returns:
        dict: Home remedy recommendation message + updated Recommendation object.
    """
    logger.info("=" * 55)
    logger.info("Node 10a: home_remedy_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    history = state.get("medical_history", MedicalHistory())
    patient = state.get("patient_profile", PatientProfile())

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a certified wellness advisor providing EVIDENCE-BASED home care guidance.
             This is for EDUCATIONAL PURPOSES ONLY — not a substitute for medical advice.

             ⚠️ CRITICAL SAFETY CONSTRAINTS (MUST FOLLOW):
             Patient allergies: {allergies}
             Current medications: {medications}
             Chronic conditions: {conditions}
             Age: {age} | Pregnancy: {pregnancy}

             RULES:
             - NEVER recommend anything that conflicts with the above
             - Flag any OTC medication that might interact with their current meds
             - Use age-appropriate dosages
             - If pregnant, be extra conservative (no NSAIDs, no high-dose vitamins)
             - Always recommend monitoring and when to escalate

             Structure your response:

             🏠 HOME REMEDIES (evidence-based):
             [2-4 specific, actionable recommendations]

             💊 OTC MEDICATIONS (if appropriate and safe):
             [Name, typical dosage, frequency — ONLY if safe for this patient]

             🍎 DIETARY GUIDANCE:
             [Foods to eat and avoid]

             💧 HYDRATION:
             [Daily intake goal, best fluids]

             😴 REST:
             [Sleep/activity recommendations]

             ⏰ EXPECTED RECOVERY TIMELINE:
             [When to expect improvement]

             🚨 ESCALATE TO DOCTOR IF:
             [Specific signs that this is getting worse — make this prominent]
             """),
            ("human", "Provide home care for: {symptoms}")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "symptoms": format_symptoms_summary(symptoms),
                "allergies": history.allergies or "None reported",
                "medications": history.current_medications or "None reported",
                "conditions": history.chronic_conditions or "None reported",
                "age": patient.age or "Unknown",
                "pregnancy": "Yes — be conservative" if patient.pregnancy_status else "No/Unknown",
            }),
            max_retries=2
        )

        home_remedy_content = response.content
        logger.info("home_remedy_node — LLM response received")

    except Exception as e:
        logger.error(f"home_remedy_node LLM failed: {e}", exc_info=True)
        home_remedy_content = (
            "🏠 HOME REMEDIES (General Safe Advice):\n"
            "• Rest as much as possible — your body heals during rest\n"
            "• Stay well hydrated (6-8 glasses of water daily)\n"
            "• Eat light, easily digestible meals\n\n"
            "⚠️ Before taking any OTC medication, check with a pharmacist "
            "given your other medications and conditions.\n\n"
            "🚨 SEE A DOCTOR IMMEDIATELY if symptoms worsen, you develop "
            "difficulty breathing, chest pain, or a very high fever (> 103°F)."
        )

    recommendation = Recommendation(
        primary_action="Home care with close monitoring",
        urgency_timeframe="Monitor for 3-5 days. See a doctor if no improvement.",
        home_remedies=[s.strip() for s in home_remedy_content.split("\n") if s.strip()][:5],
    )

    content = f"🏠 HOME CARE RECOMMENDATIONS\n\n{home_remedy_content}"
    logger.info("home_remedy_node — COMPLETE ✅")

    return {
        "recommendation": recommendation,
        "current_step": "home_remedy",
        "messages": [{"role": "assistant", "content": content}],
    }


# ══════════════════════════════════════════════
#  Node 10b: Lifestyle & Prevention Tips
# ══════════════════════════════════════════════

def lifestyle_prevention_node(state: MedicalCheckerState) -> dict:
    """
    Node 10b: Provide lifestyle modifications and prevention tips.

    Tailored to the patient's:
    - Current symptoms (recovery-focused)
    - Existing chronic conditions
    - Reported lifestyle habits

    Fallback: Generic wellness tips.

    Args:
        state: Graph state with medical history and symptoms.

    Returns:
        dict: Lifestyle/prevention tips message.
    """
    logger.info("=" * 55)
    logger.info("Node 10b: lifestyle_prevention_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    history = state.get("medical_history", MedicalHistory())

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a preventive health advisor.
             Provide PRACTICAL, SPECIFIC lifestyle and prevention guidance tailored to this patient.

             Structure your response:

             🏃 EXERCISE:
             [Safe activity levels during recovery — avoid overly generic advice]

             🥗 NUTRITION:
             [Specific foods that help this condition, foods to avoid]

             😴 SLEEP HYGIENE:
             [3-4 concrete tips for better recovery sleep]

             🧘 STRESS MANAGEMENT:
             [If relevant to their symptoms — be specific]

             💧 HYDRATION PLAN:
             [Daily target + best beverages for their condition]

             🛡️ PREVENTION TIPS:
             [How to avoid recurrence — symptom-specific]

             📅 FOLLOW-UP SCHEDULE:
             [When to self-check vs when to check in with a doctor]

             Patient lifestyle: {lifestyle}
             Chronic conditions: {conditions}
             Current symptoms: {symptoms}

             Keep advice practical, specific, and achievable. Not generic.
             """),
            ("human", "Provide lifestyle and prevention guidance.")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "lifestyle": history.lifestyle or "Not reported",
                "conditions": history.chronic_conditions or ["None reported"],
                "symptoms": [s.name for s in symptoms],
            }),
            max_retries=2
        )

        content = f"🌿 LIFESTYLE & PREVENTION TIPS\n\n{response.content}"
        logger.info("lifestyle_prevention_node — COMPLETE ✅")

        return {
            "current_step": "lifestyle_prevention",
            "messages": [{"role": "assistant", "content": content}],
        }

    except Exception as e:
        logger.error(f"lifestyle_prevention_node failed: {e}", exc_info=True)
        return {
            "current_step": "lifestyle_prevention",
            "messages": [{
                "role": "assistant",
                "content": (
                    "🌿 LIFESTYLE & PREVENTION TIPS\n\n"
                    "🏃 EXERCISE: Light activity (walking) is fine. Rest if feeling unwell.\n"
                    "🥗 NUTRITION: Eat whole foods, avoid processed meals and excess sugar.\n"
                    "😴 SLEEP: Aim for 7-9 hours. Maintain a consistent sleep schedule.\n"
                    "💧 HYDRATION: 6-8 glasses of water daily, more if feverish.\n"
                    "🛡️ PREVENTION: Wash hands frequently, avoid sick contacts.\n"
                    "📅 FOLLOW-UP: If symptoms persist beyond 5 days, see a doctor."
                )
            }],
        }
