"""LLM model configurations using Groq."""

from langchain_groq import ChatGroq
from src.config.settings import GROQ_API_KEY, DEFAULT_TEMPERATURE


def get_chat_model(model_name: str = "llama-3.3-70b-versatile", temperature: float = DEFAULT_TEMPERATURE):
    """Factory for chat models using Groq.

    Available Groq free-tier models:
    - llama-3.3-70b-versatile  (best overall)
    - llama-3.1-8b-instant     (fastest)
    - mixtral-8x7b-32768       (good for long context)
    - gemma2-9b-it             (Google's Gemma)
    """
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )
