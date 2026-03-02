"""
core/exceptions.py
==================
Custom exception hierarchy for precise error handling.
Each exception type enables specific fallback behavior in nodes/graph.
"""


class MedicalCheckerError(Exception):
    """Base exception for all Medical Symptom Checker errors."""
    pass


class NodeExecutionError(MedicalCheckerError):
    """Raised when a graph node fails during execution."""

    def __init__(self, node_name: str, message: str, original_error: Exception = None):
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"[Node:{node_name}] {message}")


class LLMError(MedicalCheckerError):
    """Raised when the LLM API call fails (after all retries)."""

    def __init__(self, message: str, retry_count: int = 0):
        self.retry_count = retry_count
        super().__init__(f"LLM Error (after {retry_count} retries): {message}")


class StateValidationError(MedicalCheckerError):
    """Raised when state data fails Pydantic validation."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"State validation failed for '{field}': {message}")


class ConfigurationError(MedicalCheckerError):
    """Raised when configuration is missing or invalid (e.g., no API key)."""
    pass


class RedFlagError(MedicalCheckerError):
    """
    Raised when critical red flags are detected requiring immediate attention.
    Caught at the graph level to ensure emergency routing.
    """

    def __init__(self, flags: list[str]):
        self.flags = flags
        super().__init__(
            f"CRITICAL RED FLAGS DETECTED ({len(flags)}): {', '.join(flags)}"
        )


class ExternalServiceError(MedicalCheckerError):
    """Raised when external services (Tavily search, etc.) fail."""

    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(f"External service '{service}' error: {message}")
