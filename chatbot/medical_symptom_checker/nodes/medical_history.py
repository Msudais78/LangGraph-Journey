"""
nodes/medical_history.py
========================
Node 6: Medical History & Allergy Collection

Collects and structures the patient's medical background: chronic conditions,
current medications, allergies, surgeries, family history, and lifestyle.
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import MedicalHistory
from medical_symptom_checker.nodes.utils import retry_llm_call, get_conversation_text

logger = get_logger(__name__)


def medical_history_node(state: MedicalCheckerState) -> dict:
    """
    Node 6: Extract and structure the patient's medical history.

    Uses LLM structured output to parse:
    - Chronic conditions, current medications, allergies
    - Past surgeries, family history, vaccination status
    - Lifestyle (smoking, alcohol, exercise, diet)

    Fallback: Returns empty MedicalHistory() and asks user to confirm.

    Args:
        state: Graph state with full conversation history.

    Returns:
        dict: Updated medical_history, question asking for confirmation/additions.
    """
    logger.info("=" * 55)
    logger.info("Node 6: medical_history_node — STARTING")
    logger.info("=" * 55)

    messages = state.get("messages", [])
    symptoms = state.get("symptoms", [])
    conversation = get_conversation_text(messages, last_n=10)
    symptoms_text = ", ".join([s.name for s in symptoms]) if symptoms else "as described"

    try:
        llm = get_llm()

        # ── Pass 1: Structured extraction ────────────────────────────
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a medical history intake specialist.
             From the conversation, extract ALL medical history mentioned:

             - chronic_conditions: ongoing conditions (diabetes, hypertension, asthma, etc.)
             - current_medications: all medications including OTC (name, dose if mentioned)
             - allergies: drug allergies, food allergies, environmental allergies
             - past_surgeries: procedures with year if mentioned
             - family_history: diseases in immediate family members
             - vaccination_status: vaccines mentioned (as dict: {{"vaccine": true/false}})
             - lifestyle: habits as dict (e.g., {{"smoker": false, "alcohol": "occasional", "exercise": "daily"}})

             Extract ONLY what is explicitly mentioned. Use empty lists/dicts for unstated fields.
             Do not infer or assume any information.
             """),
            ("human", "Conversation:\n{conversation}\n\nExtract medical history.")
        ])

        extraction_chain = extraction_prompt | llm.with_structured_output(MedicalHistory)
        history: MedicalHistory = retry_llm_call(
            lambda: extraction_chain.invoke({"conversation": conversation}),
            max_retries=2
        )

        logger.info(
            f"Medical history extracted: "
            f"{len(history.chronic_conditions)} conditions, "
            f"{len(history.current_medications)} medications, "
            f"{len(history.allergies)} allergies"
        )

        # ── Pass 2: Conversational follow-up ─────────────────────────
        followup_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a caring medical intake assistant.
             Briefly summarize the medical history you've collected, then ask
             if there's anything important they haven't mentioned yet.

             Be specific, mentioning what you DID collect so they can confirm.
             Symptoms context: {symptoms}

             Keep response under 100 words. Empathetic, professional tone.
             """),
            ("human",
             "Medical history collected:\n"
             "Conditions: {conditions}\n"
             "Medications: {meds}\n"
             "Allergies: {allergies}\n"
             "Ask if there's anything to add.")
        ])

        followup_chain = followup_prompt | llm
        followup_response = retry_llm_call(
            lambda: followup_chain.invoke({
                "symptoms": symptoms_text,
                "conditions": history.chronic_conditions or "None mentioned",
                "meds": history.current_medications or "None mentioned",
                "allergies": history.allergies or "None mentioned",
            }),
            max_retries=2
        )

        logger.info("medical_history_node — COMPLETE ✅")
        return {
            "medical_history": history,
            "current_step": "medical_history",
            "messages": [{"role": "assistant", "content": followup_response.content}],
        }

    except Exception as e:
        logger.error(f"medical_history_node failed: {e}", exc_info=True)
        return {
            "medical_history": MedicalHistory(),
            "current_step": "medical_history",
            "messages": [{
                "role": "assistant",
                "content": (
                    "Let me make note of your medical background.\n\n"
                    "To complete your assessment, please confirm:\n"
                    "1. Do you have any chronic conditions (diabetes, blood pressure, etc.)?\n"
                    "2. Are you currently taking any medications?\n"
                    "3. Do you have any known allergies?\n\n"
                    "If you've already mentioned these, we'll proceed with what I have."
                )
            }],
        }
