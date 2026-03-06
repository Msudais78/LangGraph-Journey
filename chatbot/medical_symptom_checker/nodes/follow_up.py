"""
nodes/follow_up.py
==================
Node 4: Follow-Up Question Engine

Generates targeted clinical follow-up questions when the initial
symptom description needs clarification. Implements loop control.
"""

from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.patient import PatientProfile
from medical_symptom_checker.nodes.utils import retry_llm_call, format_symptoms_summary

logger = get_logger(__name__)


def follow_up_question_node(state: MedicalCheckerState) -> dict:
    """
    Node 4: Generate targeted follow-up questions to clarify symptoms.

    Asks 2-3 focused questions based on current symptoms and patient profile.
    Tracks loop_count to avoid infinite follow-up cycles.

    Fallback: Returns a generic clarifying question if LLM fails.

    Args:
        state: Graph state with symptoms and patient profile.

    Returns:
        dict: Updated follow_up_questions, incremented loop_count, question message.
    """
    logger.info("=" * 55)
    logger.info("Node 4: follow_up_question_node — STARTING")
    logger.info("=" * 55)

    symptoms = state.get("symptoms", [])
    patient = state.get("patient_profile", PatientProfile())
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 5)

    logger.info(f"Follow-up iteration {loop_count + 1}/{max_loops}")
    logger.info(f"Current symptoms: {len(symptoms)}")

    try:
        llm = get_llm()

        symptoms_text = format_symptoms_summary(symptoms)
        patient_text = f"{patient.age}yo {patient.gender}" if patient.age else "Unknown age/gender"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an experienced triage nurse conducting a clinical symptom assessment.
             Based on the symptoms reported, generate 2-3 FOCUSED follow-up questions.

             Prioritize questions about:
             1. TIMELINE — When exactly did it start? Has it worsened?
             2. ASSOCIATED SYMPTOMS — fever, nausea, vision changes, etc.?
             3. AGGRAVATING/RELIEVING FACTORS — what makes it better or worse?
             4. RELEVANT HISTORY — recent travel, sick contacts, new medications?
             5. FUNCTIONAL IMPACT — can you walk, eat, sleep, work normally?
             6. VITAL SIGNS (if available) — any measured temperature, blood pressure?

             Patient: {patient}
             Current symptoms:
             {symptoms}

             Ask only 2-3 questions, numbered. Be empathetic but concise.
             Do NOT ask about information already provided.
             Remind: this assessment is for educational purposes only.
             """),
            ("human", "What are the most important follow-up questions for this patient?")
        ])

        chain = prompt | llm
        response = retry_llm_call(
            lambda: chain.invoke({
                "patient": patient_text,
                "symptoms": symptoms_text
            }),
            max_retries=2
        )

        new_loop_count = loop_count + 1
        logger.info(f"follow_up_question_node — COMPLETE ✅ (loop {new_loop_count}/{max_loops})")

        return {
            "follow_up_questions": [response.content],
            "loop_count": new_loop_count,
            "current_step": "follow_up",
            "messages": [{"role": "assistant", "content": response.content}],
        }

    except Exception as e:
        logger.error(f"follow_up_question_node failed: {e}", exc_info=True)
        new_loop_count = loop_count + 1
        return {
            "follow_up_questions": ["Could you please provide more detail about your main symptom?"],
            "loop_count": new_loop_count,
            "current_step": "follow_up",
            "messages": [{
                "role": "assistant",
                "content": (
                    "Could you tell me more about your symptoms?\n\n"
                    "1. How long have you had these symptoms?\n"
                    "2. How severe are they on a scale of 1-10?\n"
                    "3. Is anything making them better or worse?"
                )
            }],
        }
