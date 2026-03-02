"""
models/mental_health.py
=======================
MentalHealthScreen Pydantic model for basic mental health screening.
"""

from pydantic import BaseModel, Field


class MentalHealthScreen(BaseModel):
    """Basic mental health screening result."""

    stress_level: int = Field(
        default=0, ge=0, le=10,
        description="Self-reported stress level (0-10)"
    )
    sleep_quality: str = Field(
        default="",
        description="e.g. 'good', 'poor', 'insomnia'"
    )
    mood: str = Field(
        default="",
        description="e.g. 'stable', 'low', 'anxious'"
    )
    anxiety_indicators: bool = Field(
        default=False,
        description="Whether anxiety indicators were detected"
    )
    depression_indicators: bool = Field(
        default=False,
        description="Whether depression indicators were detected"
    )
    recommendation: str = Field(
        default="",
        description="Mental health advice or referral"
    )

    @property
    def needs_referral(self) -> bool:
        """True if professional mental health referral is recommended."""
        return self.anxiety_indicators or self.depression_indicators or self.stress_level >= 8

    @property
    def crisis_detected(self) -> bool:
        """True if crisis keywords are in the recommendation."""
        crisis_terms = ["suicidal", "self-harm", "crisis", "988", "741741"]
        return any(term in self.recommendation.lower() for term in crisis_terms)
