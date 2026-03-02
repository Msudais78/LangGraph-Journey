"""
core/config.py
==============
Centralized configuration management.
Loads .env, initializes Gemini 2.5 Flash LLM, and exposes typed app settings.
"""

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from langchain_google_genai import ChatGoogleGenerativeAI

# ──────────────────────────────────────────────
#  Project Paths
# ──────────────────────────────────────────────
# Package: e:\AgenticAI\medical_symptom_checker\core\config.py
# Project root: e:\AgenticAI
PROJECT_ROOT = Path(__file__).parent.parent.parent   # e:\AgenticAI
PACKAGE_ROOT = Path(__file__).parent.parent           # ...medical_symptom_checker\

# Load .env from e:\AgenticAI\.env
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=False)

DATA_DIR = PACKAGE_ROOT / "data"
LOGS_DIR = PACKAGE_ROOT / "logs"


# ──────────────────────────────────────────────
#  Typed Settings via pydantic-settings
# ──────────────────────────────────────────────
class AppSettings(BaseSettings):
    """
    Typed, validated application settings.
    All values read from environment / .env file.
    """

    # API Keys
    google_api_key: str = ""
    tavily_api_key: str = ""

    # LLM Config
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_timeout: int = 60

    # App Config
    max_follow_up_loops: int = 5
    log_level: str = "INFO"
    log_file: str = "logs/medical_checker.log"
    enable_streaming: bool = True

    # LangSmith (Optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "medical-symptom-checker"

    class Config:
        env_file = str(ENV_PATH)
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# ──────────────────────────────────────────────
#  Singletons
# ──────────────────────────────────────────────
@lru_cache()
def get_settings() -> AppSettings:
    """Return cached settings singleton."""
    return AppSettings()


settings: AppSettings = get_settings()


@lru_cache()
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return cached Gemini 2.5 Flash LLM instance.

    Usage:
        from medical_symptom_checker.core.config import get_llm
        llm = get_llm()
        response = llm.invoke("Hello")
    """
    _settings = get_settings()

    if not _settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Please add it to your .env file:\n"
            f"  File: {ENV_PATH}\n"
            "  Example: GOOGLE_API_KEY=AIza..."
        )

    return ChatGoogleGenerativeAI(
        model=_settings.llm_model,
        temperature=_settings.llm_temperature,
        max_output_tokens=_settings.llm_max_tokens,
        timeout=_settings.llm_timeout,
        google_api_key=_settings.google_api_key,
        convert_system_message_to_human=True,  # Gemini requirement
    )


# ──────────────────────────────────────────────
#  Path Helpers
# ──────────────────────────────────────────────
def get_data_path(filename: str) -> Path:
    """Get path to a data file in the data/ directory."""
    return DATA_DIR / filename


def ensure_dirs() -> None:
    """Create required runtime directories if they don't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
