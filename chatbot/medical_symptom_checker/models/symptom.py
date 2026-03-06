"""
models/symptom.py
=================
Symptom and VitalSigns Pydantic models with validators and computed properties.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Symptom(BaseModel):
    """Individual symptom with full clinical detail."""

    name: str = Field(..., description="Symptom name, e.g. 'headache'")
    body_location: str = Field(default="", description="Anatomical location, e.g. 'frontal'")
    duration: str = Field(default="", description="How long, e.g. '3 days', '2 hours'")
    intensity: int = Field(default=0, ge=0, le=10, description="Pain scale 0-10")
    frequency: str = Field(default="", description="e.g. 'constant', 'intermittent'")
    triggers: list[str] = Field(default_factory=list, description="What makes it worse")
    relieving_factors: list[str] = Field(default_factory=list, description="What makes it better")
    associated_symptoms: list[str] = Field(default_factory=list, description="Co-occurring symptoms")
    onset: str = Field(default="", description="e.g. 'sudden', 'gradual'")

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Ensure symptom name is lowercase and stripped."""
        return str(v).strip().lower() if v else v

    @field_validator("onset", mode="before")
    @classmethod
    def normalize_onset(cls, v: str) -> str:
        """Normalize onset terminology."""
        if not v:
            return ""
        v = str(v).strip().lower()
        sudden_terms = {"sudden", "acute", "abrupt", "rapid", "immediate", "instant"}
        gradual_terms = {"gradual", "slow", "progressive", "insidious", "chronic"}
        if v in sudden_terms:
            return "sudden"
        if v in gradual_terms:
            return "gradual"
        return v

    # ── Computed Clinical Properties ──

    @property
    def is_acute(self) -> bool:
        """True if symptom has sudden/acute onset or very short duration."""
        acute_keywords = ["minute", "hour", "sudden", "just started", "today", "this morning"]
        return (
            self.onset == "sudden"
            or any(kw in self.duration.lower() for kw in acute_keywords)
        )

    @property
    def is_severe(self) -> bool:
        """True if intensity is 7 or above (clinical threshold for severe)."""
        return self.intensity >= 7

    @property
    def is_moderate(self) -> bool:
        """True if intensity is 4-6."""
        return 4 <= self.intensity <= 6

    @property
    def clinical_summary(self) -> str:
        """Short clinical summary for logging/display."""
        parts = [self.name]
        if self.intensity:
            parts.append(f"intensity {self.intensity}/10")
        if self.duration:
            parts.append(f"for {self.duration}")
        if self.onset:
            parts.append(f"onset: {self.onset}")
        return " | ".join(parts)


class VitalSigns(BaseModel):
    """
    Optional vital sign readings — all fields are Optional since
    most patients won't have formal measurements.
    """

    temperature_f: Optional[float] = Field(
        default=None, ge=90.0, le=115.0,
        description="Body temperature in Fahrenheit"
    )
    heart_rate_bpm: Optional[int] = Field(
        default=None, ge=20, le=300,
        description="Heart rate in beats per minute"
    )
    blood_pressure: Optional[str] = Field(
        default=None,
        description="Blood pressure as 'systolic/diastolic', e.g. '120/80'"
    )
    respiratory_rate: Optional[int] = Field(
        default=None, ge=4, le=60,
        description="Breaths per minute"
    )
    oxygen_saturation: Optional[float] = Field(
        default=None, ge=50.0, le=100.0,
        description="SpO2 percentage"
    )

    @field_validator("blood_pressure", mode="before")
    @classmethod
    def validate_bp(cls, v: Optional[str]) -> Optional[str]:
        """Validate blood pressure format (e.g., '120/80')."""
        if v is None or str(v).strip() == "":
            return None
        v = str(v).strip()
        if "/" in v:
            parts = v.split("/")
            if len(parts) == 2:
                try:
                    systolic = int(parts[0].strip())
                    diastolic = int(parts[1].strip())
                    if 50 <= systolic <= 300 and 30 <= diastolic <= 200:
                        return f"{systolic}/{diastolic}"
                except ValueError:
                    pass
        return None  # Invalid format → None

    # ── Computed Clinical Properties ──

    @property
    def has_fever(self) -> bool:
        """True if temperature ≥ 100.4°F (38°C)."""
        return self.temperature_f is not None and self.temperature_f >= 100.4

    @property
    def has_high_fever(self) -> bool:
        """True if temperature ≥ 103°F (39.4°C)."""
        return self.temperature_f is not None and self.temperature_f >= 103.0

    @property
    def has_low_oxygen(self) -> bool:
        """True if SpO2 < 94% (clinically concerning threshold)."""
        return self.oxygen_saturation is not None and self.oxygen_saturation < 94.0

    @property
    def has_abnormal_hr(self) -> bool:
        """True if HR is bradycardic (<50) or tachycardic (>120)."""
        if self.heart_rate_bpm is None:
            return False
        return self.heart_rate_bpm < 50 or self.heart_rate_bpm > 120

    @property
    def any_abnormal(self) -> bool:
        """True if any vital sign is outside normal range."""
        return (
            self.has_fever
            or self.has_low_oxygen
            or self.has_abnormal_hr
        )
