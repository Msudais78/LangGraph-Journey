"""Application settings loaded from environment."""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///omniassist.db")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_RESEARCH_SOURCES = 5
DEFAULT_PERSONA = "professional"
DEFAULT_LANGUAGE = "en"
