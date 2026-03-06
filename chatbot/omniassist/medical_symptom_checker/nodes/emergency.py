"""
nodes/emergency.py
==================
Nodes 8a & 8b: Emergency Guidance + Nearest Hospital Finder

The SEVERE/EMERGENCY pathway.
8a: Provides immediate emergency action steps.
8b: Finds nearby emergency facilities via Tavily search.
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm, get_settings
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.core.exceptions import ExternalServiceError
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile
from medical_symptom_checker.models.assessment import SeverityAssessment, Recommendation
from medical_symptom_checker.nodes.utils import retry_llm_call

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Node 8a: Emergency Guidance
# ══════════════════════════════════════════════

def emergency_guidance_node(state: MedicalCheckerState) -> dict:
    """
    Node 8a: Provide clear, immediate emergency guidance.

    Tells the patient:
    - What to do RIGHT NOW
    - Emergency numbers to call
    - What to do while waiting for help
    - What NOT to do
    - Information to tell emergency services

    Fallback: Static emergency template if LLM fails.

    Args:
        state: Graph state with symptoms and severity assessment.

    Returns:
        dict: Emergency recommendation + critical action message.
    """
    logger.info("=" * 55)
    logger.info("Node 8a: emergency_guidance_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    severity = state.get("severity_assessment", SeverityAssessment())
    patient = state.get("patient_profile", PatientProfile())

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an emergency medical guidance system providing URGENT educational guidance.
             This patient has been assessed as SEVERE or EMERGENCY level. Provide clear, actionable steps.

             Structure your response EXACTLY as:

             🚨 IMMEDIATE ACTIONS (do these RIGHT NOW):
             [2-3 bullet points — most critical actions]

             📞 CALL FOR HELP:
             [Emergency numbers and when to call each]

             ⏳ WHILE WAITING FOR HELP:
             [What patient/bystanders should do]

             ❌ DO NOT:
             [Dangerous actions to avoid]

             📝 TELL EMERGENCY SERVICES:
             [Key information to communicate]

             ⚠️ ALWAYS REMIND: This is educational guidance. Real emergency services should be contacted immediately.

             Red flags detected: {red_flags}
             Symptoms: {symptoms}
             Possible conditions: {conditions}
             Patient: {patient}

             Use SIMPLE, CLEAR language. Assume the reader may be panicking.
             Every word matters in an emergency.
             """),
            ("human", "Provide emergency guidance.")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "red_flags": severity.red_flags_detected or ["High severity symptoms"],
                "symptoms": [s.name for s in symptoms],
                "conditions": severity.top_conditions,
                "patient": patient.display_name,
            }),
            max_retries=2
        )

        guidance_content = response.content
        logger.info("emergency_guidance_node — COMPLETE ✅")

    except Exception as e:
        logger.error(f"emergency_guidance_node LLM failed: {e}", exc_info=True)
        # Static emergency fallback
        guidance_content = (
            "🚨 IMMEDIATE ACTIONS:\n"
            "• Call 911 (US) or your local emergency number IMMEDIATELY\n"
            "• Do not drive yourself to the hospital if symptoms are severe\n"
            "• Stay calm and stay with someone if possible\n\n"
            "📞 EMERGENCY CONTACTS:\n"
            "• 911 — Emergency Services\n"
            "• Poison Control: 1-800-222-1222\n"
            "• 988 — Suicide & Crisis Lifeline\n\n"
            "⏳ WHILE WAITING:\n"
            "• Stay still and comfortable\n"
            "• Loosen any tight clothing\n"
            "• Do not eat or drink anything\n\n"
            "⚠️ This is educational guidance. Call emergency services now."
        )

    emergency_msg = f"🚨🚨🚨  EMERGENCY ALERT  🚨🚨🚨\n\n{guidance_content}"

    emergency_recommendation = Recommendation(
        primary_action="SEEK EMERGENCY CARE IMMEDIATELY",
        urgency_timeframe="NOW — Do not delay. Call 911 or go to the nearest emergency room.",
        emergency_numbers=["911 (US Emergency)", "Poison Control: 1-800-222-1222",
                           "988 Suicide & Crisis Lifeline"],
    )

    return {
        "recommendation": emergency_recommendation,
        "current_step": "emergency_guidance",
        "messages": [{"role": "assistant", "content": emergency_msg}],
    }


# ══════════════════════════════════════════════
#  Node 8b: Nearest Hospital Finder
# ══════════════════════════════════════════════

def nearest_hospital_node(state: MedicalCheckerState) -> dict:
    """
    Node 8b: Find nearest emergency facilities using Tavily search.

    Only runs if patient location is provided.
    Fallback: Directs to Google Maps if no location or search fails.

    Args:
        state: Graph state with patient profile (location field).

    Returns:
        dict: Updated recommendation with nearest_facilities + hospital message.
    """
    logger.info("=" * 55)
    logger.info("Node 8b: nearest_hospital_node — STARTING")
    logger.info("=" * 55)

    patient = state.get("patient_profile", PatientProfile())
    current_rec = state.get("recommendation", Recommendation())
    location = patient.location

    facilities = []
    hospital_msg = ""

    if location:
        try:
            settings = get_settings()
            if not settings.tavily_api_key:
                raise ExternalServiceError("Tavily", "TAVILY_API_KEY not configured")

            from langchain_community.tools.tavily_search import TavilySearchResults
            search_tool = TavilySearchResults(
                max_results=3,
                tavily_api_key=settings.tavily_api_key
            )

            logger.info(f"Searching for emergency facilities near: {location}")
            results = search_tool.invoke(f"emergency room hospital near {location}")

            for r in results:
                if isinstance(r, dict):
                    facilities.append({
                        "name": r.get("title", "Emergency Facility"),
                        "info": r.get("content", "")[:200],
                        "url": r.get("url", ""),
                    })

            logger.info(f"Found {len(facilities)} facility result(s)")

            if facilities:
                hospital_msg = (
                    f"🏥 NEAREST EMERGENCY FACILITIES NEAR {location.upper()}:\n\n"
                    + "\n".join([
                        f"  {i+1}. {f['name']}\n     {f['info'][:120]}..."
                        for i, f in enumerate(facilities)
                    ])
                    + "\n\n📍 Also search 'Emergency Room near me' on Google Maps for current wait times."
                )
            else:
                raise ExternalServiceError("Tavily", "No results returned")

        except Exception as e:
            logger.warning(f"Hospital search failed: {e}")
            facilities = []

    if not facilities:
        hospital_msg = (
            "🏥 TO FIND THE NEAREST EMERGENCY FACILITY:\n\n"
            "• 📍 Google Maps: Search 'Emergency Room near me'\n"
            "• 📱 Apple Maps: Search 'hospital'\n"
            "• 🌐 Er.FindMechanic.com — real-time ER wait times\n"
            "• 📞 Call 911 and they will dispatch help or guide you\n\n"
            "⏱️ Choose the option that gets you help FASTEST."
        )

    # Update recommendation with facilities
    try:
        current_rec.nearest_facilities = facilities
    except Exception:
        pass  # If recommendation is immutable, skip update

    logger.info("nearest_hospital_node — COMPLETE ✅")
    return {
        "recommendation": current_rec,
        "current_step": "hospital_finder",
        "messages": [{"role": "assistant", "content": hospital_msg}],
    }
