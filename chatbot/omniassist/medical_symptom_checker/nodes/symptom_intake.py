"""
nodes/symptom_intake.py
=======================
Node 3: Multi-Turn Symptom Intake

Extracts structured symptom data from the patient's free-text
description using a dual-LLM approach: conversational response + structured extraction.
"""

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from medical_symptom_checker.core.config import get_llm
from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState
from medical_symptom_checker.models.symptom import Symptom
from medical_symptom_checker.nodes.utils import retry_llm_call, get_conversation_text, format_symptoms_summary

logger = get_logger(__name__)


class SymptomList(BaseModel):
    """Structured output schema for symptom parsing."""
    symptoms: list[Symptom]
    needs_clarification: bool = False
    clarification_question: str = ""


def symptom_intake_node(state: MedicalCheckerState) -> dict:
    """
    Node 3: Parse and structure the patient's symptom descriptions.

    Two LLM passes:
    1. Conversational: empathetic response + any clarifying questions
    2. Structured: extract Symptom objects from the full conversation

    Fallback: Keyword-based symptom extraction if LLM fails.

    Args:
        state: Graph state with messages containing symptom descriptions.

    Returns:
        dict: Updated symptoms list, needs_more_info flag, response message.
    """
    logger.info("=" * 55)
    logger.info("Node 3: symptom_intake_node — STARTING")
    logger.info("=" * 55)

    messages = state.get("messages", [])
    patient = state.get("patient_profile")
    conversation = get_conversation_text(messages, last_n=8)

    try:
        llm = get_llm()

        # ── Pass 1: Conversational response ──────────────────────────
        conversational_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an empathetic medical intake specialist.
             Acknowledge the patient's symptoms with care and brief follow-up.

             Important: Be empathetic but efficient. Acknowledge what they've told you,
             then ask the ONE most important clarifying question if anything is unclear.
             If you have enough information, confirm what you heard and proceed.

             Keep your response concise (3-5 sentences max).
             Always remind: this is educational only.
             """),
            ("human", "Conversation so far:\n{conversation}")
        ])

        conv_chain = conversational_prompt | llm
        conv_response = retry_llm_call(
            lambda: conv_chain.invoke({"conversation": conversation}),
            max_retries=2
        )

        # ── Pass 2: Structured symptom extraction ────────────────────
        parser_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a clinical symptom data extractor.
             From the conversation, extract ALL symptoms mentioned with their details.

             For each symptom extract:
             - name: lowercase symptom name (required)
             - body_location: where on the body (e.g., 'frontal', 'chest', 'lower back')
             - duration: how long (e.g., '3 days', '2 hours', 'a week')
             - intensity: 0-10 pain/severity scale (estimate if not stated)
             - frequency: 'constant', 'intermittent', 'episodic', 'getting worse'
             - onset: 'sudden' or 'gradual'
             - triggers: what makes it worse (list)
             - relieving_factors: what makes it better (list)
             - associated_symptoms: other symptoms mentioned alongside this one (list)

             Also determine:
             - needs_clarification: true only if critical information is completely missing
             - clarification_question: the single most important missing piece of info

             Extract ONLY what's explicitly mentioned. Use sensible defaults for unstated fields.
             """),
            ("human", "Conversation:\n{conversation}\n\nExtract structured symptoms.")
        ])

        parser_chain = parser_prompt | llm.with_structured_output(SymptomList)
        parsed: SymptomList = retry_llm_call(
            lambda: parser_chain.invoke({"conversation": conversation}),
            max_retries=2
        )

        logger.info(
            f"Extracted {len(parsed.symptoms)} symptom(s). "
            f"Needs clarification: {parsed.needs_clarification}"
        )
        for s in parsed.symptoms:
            logger.debug(f"  Symptom: {s.clinical_summary}")

        logger.info("symptom_intake_node — COMPLETE ✅")
        return {
            "symptoms": parsed.symptoms,
            "needs_more_info": parsed.needs_clarification,
            "symptom_collection_complete": not parsed.needs_clarification,
            "current_step": "symptom_intake",
            "messages": [{"role": "assistant", "content": conv_response.content}],
        }

    except Exception as e:
        logger.error(f"symptom_intake_node failed: {e}", exc_info=True)

        # ── Keyword-based fallback ───────────────────────────────────
        logger.warning("Using keyword-based symptom extraction as fallback")
        all_text = " ".join([
            m.content if hasattr(m, "content") else m.get("content", "")
            for m in messages
        ]).lower()

        # Basic keyword matching
        COMMON_SYMPTOMS = [
            "headache", "fever", "cough", "pain", "nausea", "vomiting",
            "dizziness", "fatigue", "shortness of breath", "chest pain",
            "stomach ache", "sore throat", "runny nose", "rash",
        ]
        found_symptoms = [
            Symptom(name=kw) for kw in COMMON_SYMPTOMS if kw in all_text
        ]

        if not found_symptoms:
            found_symptoms = [Symptom(name="unspecified complaint", intensity=0)]

        logger.warning(f"Fallback extracted {len(found_symptoms)} symptom keyword(s)")

        return {
            "symptoms": found_symptoms,
            "needs_more_info": True,
            "symptom_collection_complete": False,
            "current_step": "symptom_intake",
            "messages": [{
                "role": "assistant",
                "content": (
                    "I've noted your symptoms. To make a better assessment, "
                    "could you describe your main symptom and rate its severity on a scale of 1-10?"
                )
            }],
        }
