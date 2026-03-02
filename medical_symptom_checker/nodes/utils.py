"""
nodes/utils.py
==============
Shared utilities for all node implementations.
Provides data loaders, LLM retry logic, and output formatters.
"""

import json
import time
from pathlib import Path

from medical_symptom_checker.core.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Data Loaders (with fallbacks)
# ══════════════════════════════════════════════

def load_red_flags() -> dict:
    """Load red flag keywords from data/red_flags.json with hardcoded fallback."""
    try:
        from medical_symptom_checker.core.config import get_data_path
        path = get_data_path("red_flags.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"Loaded {len(data)} red flag categories from file")
            return data
    except Exception as e:
        logger.warning(f"Could not load red_flags.json: {e}. Using fallback.")
        return {
            "chest_pain": ["chest pain", "chest tightness", "crushing chest", "chest pressure"],
            "breathing": ["difficulty breathing", "shortness of breath", "can't breathe"],
            "stroke_signs": ["face drooping", "arm weakness", "speech difficulty",
                             "sudden numbness", "sudden confusion"],
            "severe_bleeding": ["uncontrollable bleeding", "coughing blood",
                                "vomiting blood", "blood in stool"],
            "consciousness": ["fainting", "loss of consciousness", "unresponsive",
                              "seizure", "convulsion"],
            "anaphylaxis": ["throat swelling", "severe allergic reaction", "can't swallow"],
            "head_injury": ["severe head injury", "head trauma"],
            "suicidal": ["suicidal thoughts", "want to end my life", "self harm"],
            "overdose": ["overdose", "took too many pills", "poisoning"],
            "severe_pain": ["worst pain of my life", "10/10 pain", "unbearable pain"],
            "high_fever": ["fever above 104", "fever 40"],
            "pregnancy_emergency": ["heavy vaginal bleeding pregnant",
                                    "severe abdominal pain pregnant"],
        }


def load_specialist_map() -> dict:
    """Load specialist mapping from data/specialist_map.json with hardcoded fallback."""
    try:
        from medical_symptom_checker.core.config import get_data_path
        path = get_data_path("specialist_map.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"Loaded {len(data)} specialist mappings from file")
            return data
    except Exception as e:
        logger.warning(f"Could not load specialist_map.json: {e}. Using fallback.")
        return {
            "head": "Neurologist", "heart": "Cardiologist",
            "chest": "Pulmonologist / Cardiologist", "stomach": "Gastroenterologist",
            "abdomen": "Gastroenterologist", "skin": "Dermatologist",
            "bone": "Orthopedist", "joint": "Rheumatologist",
            "eye": "Ophthalmologist", "ear": "ENT (Otolaryngologist)",
            "throat": "ENT (Otolaryngologist)", "mental": "Psychiatrist / Psychologist",
            "kidney": "Nephrologist", "urinary": "Urologist",
            "allergy": "Allergist / Immunologist", "general": "Primary Care Physician",
        }


def load_disclaimer() -> str:
    """Load disclaimer text from data/disclaimer.txt with hardcoded fallback."""
    try:
        from medical_symptom_checker.core.config import get_data_path
        path = get_data_path("disclaimer.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not load disclaimer.txt: {e}. Using inline fallback.")
        return (
            "⚠️ IMPORTANT DISCLAIMER: This tool is for EDUCATIONAL PURPOSES ONLY. "
            "It does NOT provide medical advice, diagnosis, or treatment. "
            "Always consult a licensed healthcare professional. "
            "In emergencies, call 911 immediately."
        )


# ══════════════════════════════════════════════
#  LLM Retry Wrapper
# ══════════════════════════════════════════════

def retry_llm_call(func, max_retries: int = 2, base_delay: float = 1.0):
    """
    Execute an LLM call with automatic retry on failure.

    Uses exponential backoff (delay doubles each retry).

    Args:
        func: A zero-argument callable that performs the LLM call.
        max_retries: Maximum number of retry attempts (default: 2).
        base_delay: Initial delay in seconds before first retry.

    Returns:
        The result of func() on success.

    Raises:
        The last exception if all retries exhausted.

    Usage:
        result = retry_llm_call(lambda: chain.invoke(inputs))
    """
    last_error = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{max_retries + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:100]}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2.0  # Exponential backoff
            else:
                logger.error(
                    f"LLM call failed after {max_retries + 1} attempts: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

    raise last_error


# ══════════════════════════════════════════════
#  Output Formatters
# ══════════════════════════════════════════════

SEPARATOR_THIN = "─" * 55
SEPARATOR_THICK = "═" * 55


def format_section_header(title: str, emoji: str = "📋") -> str:
    """Format a consistent, readable section header."""
    return f"\n{SEPARATOR_THICK}\n{emoji} {title}\n{SEPARATOR_THICK}"


def format_symptoms_summary(symptoms: list) -> str:
    """Format a symptoms list into a human-readable summary for prompts."""
    if not symptoms:
        return "No symptoms recorded."
    lines = []
    for s in symptoms:
        line = f"  • {s.name}"
        if s.intensity:
            line += f" (intensity: {s.intensity}/10"
        if s.duration:
            line += f", duration: {s.duration}"
        if s.body_location:
            line += f", location: {s.body_location}"
        if s.onset:
            line += f", onset: {s.onset}"
        if s.intensity:
            line += ")"
        lines.append(line)
    return "\n".join(lines)


def format_patient_summary(patient) -> str:
    """Format a patient profile into a readable string for prompts."""
    parts = []
    if patient.name:
        parts.append(patient.name)
    if patient.age:
        parts.append(f"{patient.age} years old")
    if patient.gender:
        parts.append(patient.gender)
    if patient.location:
        parts.append(f"from {patient.location}")
    return ", ".join(parts) if parts else "Unknown patient"


def get_conversation_text(messages: list, last_n: int = 10) -> str:
    """Extract recent conversation as a formatted text block for prompts."""
    recent = messages[-last_n:] if len(messages) > last_n else messages
    lines = []
    for m in recent:
        if hasattr(m, "type") and hasattr(m, "content"):
            role = "User" if m.type == "human" else "Assistant"
            lines.append(f"{role}: {m.content}")
        elif isinstance(m, dict):
            role = m.get("role", "unknown").capitalize()
            lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines) if lines else "No conversation history."
