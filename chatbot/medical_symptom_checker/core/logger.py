"""
core/logger.py
==============
Centralized logging configuration.
Provides structured, leveled logging with file + console output.
"""

import logging
import sys
from pathlib import Path


# ──────────────────────────────────────────────
#  Log Format Constants
# ──────────────────────────────────────────────
CONSOLE_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s"
FILE_FORMAT    = "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(funcName)s:%(lineno)d │ %(message)s"
DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

# Cache of initialized loggers to avoid duplicate handlers
_initialized: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Attaches a console handler (INFO+) and a file handler (DEBUG+) on first call.
    Subsequent calls for the same name return the cached logger.

    Args:
        name: Logger name — typically pass __name__ from the calling module.

    Returns:
        Configured logging.Logger instance.

    Usage:
        logger = get_logger(__name__)
        logger.info("Node executed successfully")
        logger.error("LLM call failed", exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers on repeated imports
    if name in _initialized:
        return logger

    _initialized.add(name)

    # Determine log level from settings (lazy import to avoid circular deps)
    try:
        from medical_symptom_checker.core.config import settings, LOGS_DIR
        log_level_str = settings.log_level.upper()
        logs_dir = LOGS_DIR
        log_filename = Path(settings.log_file).name
    except Exception:
        log_level_str = "INFO"
        logs_dir = Path(__file__).parent.parent / "logs"
        log_filename = "medical_checker.log"

    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(logging.DEBUG)  # Let handlers filter

    # ── Console Handler (INFO and above) ──
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

    # ── File Handler (DEBUG and above) ──
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = logs_dir / log_filename
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create file log handler: {e}")

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger
