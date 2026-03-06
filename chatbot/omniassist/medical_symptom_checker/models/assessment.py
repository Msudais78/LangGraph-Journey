"""
models/assessment.py
====================
SeverityAssessment and Recommendation Pydantic models.
"""

from typing import Literal
from pydantic import BaseModel, Field


class SeverityAssessment(BaseModel):
    """Classification result from the severity_classifier_node."""

    level: Literal["mild", "moderate", "severe", "emergency"] = "mild"
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="LLM confidence in the classification (0.0-1.0)"
    )
    triage_score: int = Field(
        default=5, ge=1, le=5,
        description="Emergency Severity Index (ESI) 1=most urgent, 5=least urgent"
    )
    reasoning: str = Field(
        default="",
        description="LLM's clinical reasoning for the classification"
    )
    red_flags_detected: list[str] = Field(
        default_factory=list,
        description="List of red flag descriptions found"
    )
    possible_conditions: list[dict] = Field(
        default_factory=list,
        description="List of dicts: [{'name': '...', 'probability': 0.8}]"
    )

    # ── Computed Properties ──

    @property
    def is_emergency(self) -> bool:
        """True if severity level is 'emergency' or 'severe'."""
        return self.level in ("severe", "emergency")

    @property
    def has_red_flags(self) -> bool:
        """True if any red flags were detected."""
        return len(self.red_flags_detected) > 0

    @property
    def severity_emoji(self) -> str:
        """Emoji representation of severity level."""
        return {
            "mild": "🟢",
            "moderate": "🟡",
            "severe": "🔴",
            "emergency": "🚨",
        }.get(self.level, "⚪")

    @property
    def top_conditions(self) -> list[str]:
        """Get top 3 possible conditions as formatted strings."""
        result = []
        for c in self.possible_conditions[:3]:
            name = c.get("name", "Unknown")
            prob = c.get("probability", "N/A")
            if isinstance(prob, float):
                result.append(f"{name} ({prob:.0%})")
            else:
                result.append(f"{name} ({prob})")
        return result


class Recommendation(BaseModel):
    """Final recommendation output aggregated from pathway nodes."""

    primary_action: str = Field(default="", description="Main action the patient should take")
    home_remedies: list[str] = Field(default_factory=list, description="Evidence-based home care")
    otc_medications: list[str] = Field(default_factory=list, description="Safe OTC medication suggestions")
    specialist_type: str = Field(default="", description="Recommended medical specialist")
    urgency_timeframe: str = Field(default="", description="e.g. 'immediately', 'within 24h'")
    lifestyle_tips: list[str] = Field(default_factory=list, description="Diet, exercise, sleep tips")
    drug_interactions_warning: list[str] = Field(
        default_factory=list,
        description="Potential drug interaction warnings"
    )
    follow_up_date: str = Field(default="", description="Suggested follow-up timeframe")
    emergency_numbers: list[str] = Field(
        default_factory=list,
        description="Emergency contact numbers"
    )
    nearest_facilities: list[dict] = Field(
        default_factory=list,
        description="Nearby hospital/clinic info: [{'name': '...', 'info': '...'}]"
    )
