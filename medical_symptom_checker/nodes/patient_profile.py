"""
nodes/patient_profile.py
=========================
Node 2: Patient Profile Collector

Extracts structured patient demographics from the conversation
using Gemini's structured output capability.
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile
from medical_symptom_checker.nodes.utils import retry_llm_call, get_conversation_text

logger = get_logger(__name__)


def patient_profile_node(state: MedicalCheckerState) -> dict:
    """
    Node 2: Extract patient demographics from conversation.

    Uses LLM structured output to parse: name, age, gender,
    weight, height, blood type, pregnancy status, location.

    Fallback: Returns empty PatientProfile() if LLM fails.

    Args:
        state: Graph state with messages from greeting + user input.

    Returns:
        dict: Updated patient_profile + confirmation message.
    """
    logger.info("=" * 55)
    logger.info("Node 2: patient_profile_node — STARTING")
    logger.info("=" * 55)

    messages = state.get("messages", [])
    conversation = get_conversation_text(messages, last_n=5)

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a medical intake specialist extracting patient demographics.
             From the conversation below, extract:
             - name (string, empty if not mentioned)
             - age (integer 0-150, 0 if not mentioned)
             - gender (male/female/non-binary/other/unspecified)
             - weight_kg (float, 0.0 if not mentioned)
             - height_cm (float, 0.0 if not mentioned)
             - blood_type (A+/A-/B+/B-/AB+/AB-/O+/O-, empty if not mentioned)
             - pregnancy_status (true/false/null — only set if female and mentioned)
             - location (city, state, or country — empty if not mentioned)

             Extract ONLY what is explicitly stated. Never assume or infer.
             If a field is not mentioned, use the default (empty string or 0).
             """),
            ("human", "Conversation:\n{conversation}\n\nExtract patient profile.")
        ])

        chain = prompt | llm.with_structured_output(PatientProfile)

        profile: PatientProfile = retry_llm_call(
            lambda: chain.invoke({"conversation": conversation}),
            max_retries=2
        )

        logger.info(f"Profile extracted: {profile.display_name}")

        # Build confirmation message
        profile_summary = []
        if profile.name:
            profile_summary.append(f"Name: {profile.name}")
        if profile.age:
            profile_summary.append(f"Age: {profile.age}")
        if profile.gender:
            profile_summary.append(f"Gender: {profile.gender}")
        if profile.location:
            profile_summary.append(f"Location: {profile.location}")

        summary_str = " | ".join(profile_summary) if profile_summary else "Profile noted"

        confirmation = (
            f"✅ Profile recorded: {summary_str}\n\n"
            "Now, please describe your symptoms in detail. For each symptom, it helps to include:\n"
            "  • How long you've had it\n"
            "  • How severe it is (1-10)\n"
            "  • Whether it's constant or comes and goes\n"
            "  • What makes it better or worse"
        )

        logger.info("patient_profile_node — COMPLETE ✅")
        return {
            "patient_profile": profile,
            "current_step": "patient_profile",
            "messages": [{"role": "assistant", "content": confirmation}],
        }

    except Exception as e:
        logger.error(f"patient_profile_node failed: {e}", exc_info=True)
        # Graceful fallback — proceed with empty profile
        return {
            "patient_profile": PatientProfile(),
            "current_step": "patient_profile",
            "messages": [{
                "role": "assistant",
                "content": (
                    "⚠️ I had some trouble reading your profile details, but let's continue.\n\n"
                    "Please describe your symptoms — what are you experiencing, "
                    "how long have you had them, and how severe are they (1-10)?"
                )
            }],
        }
