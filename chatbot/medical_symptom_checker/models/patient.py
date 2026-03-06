"""
models/patient.py
=================
PatientProfile and MedicalHistory Pydantic models with field validators.
"""

from typing import Optional, ClassVar
from pydantic import BaseModel, Field, field_validator


class PatientProfile(BaseModel):
    """Demographics & baseline info collected during intake."""

    name: str = ""
    age: int = Field(default=0, ge=0, le=150, description="Patient age in years")
    gender: str = ""
    weight_kg: float = Field(default=0.0, ge=0, description="Body weight in kg")
    height_cm: float = Field(default=0.0, ge=0, description="Height in centimeters")
    blood_type: str = ""
    pregnancy_status: Optional[bool] = None
    location: str = ""  # Used for hospital search

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, v: str) -> str:
        """Normalize gender input to standard terms."""
        if not v:
            return v
        v = str(v).strip().lower()
        gender_map = {
            "m": "male", "f": "female",
            "male": "male", "female": "female",
            "nb": "non-binary", "non-binary": "non-binary",
            "nonbinary": "non-binary", "other": "other",
            "prefer not to say": "unspecified",
        }
        return gender_map.get(v, v)

    @field_validator("blood_type", mode="before")
    @classmethod
    def normalize_blood_type(cls, v: str) -> str:
        """Validate and normalize blood type format."""
        if not v:
            return ""
        v = str(v).strip().upper()
        valid_types = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
        return v if v in valid_types else ""

    @property
    def is_elderly(self) -> bool:
        """True if patient is 65 or older."""
        return self.age >= 65

    @property
    def is_pediatric(self) -> bool:
        """True if patient is under 18."""
        return 0 < self.age < 18

    @property
    def is_high_risk_age(self) -> bool:
        """True if age is > 65 or < 5 (higher clinical risk groups)."""
        return self.age > 65 or (0 < self.age < 5)

    @property
    def display_name(self) -> str:
        """Human-readable label for logs and UI."""
        parts = []
        if self.name:
            parts.append(self.name)
        if self.age:
            parts.append(f"{self.age}yo")
        if self.gender:
            parts.append(self.gender)
        return ", ".join(parts) if parts else "Unknown patient"


class MedicalHistory(BaseModel):
    """Patient's full medical background."""

    chronic_conditions: list[str] = Field(
        default_factory=list,
        description="e.g. ['diabetes', 'hypertension']"
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="e.g. ['metformin 500mg', 'lisinopril 10mg']"
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="Drugs, foods, or environmental allergies"
    )
    past_surgeries: list[str] = Field(
        default_factory=list,
        description="e.g. ['appendectomy 2019']"
    )
    family_history: list[str] = Field(
        default_factory=list,
        description="e.g. ['heart disease', 'diabetes']"
    )
    vaccination_status: dict = Field(
        default_factory=dict,
        description="e.g. {'covid': True, 'flu': True}"
    )
    lifestyle: dict = Field(
        default_factory=dict,
        description="e.g. {'smoker': False, 'alcohol': 'occasional', 'exercise': 'weekly'}"
    )

    @field_validator("allergies", "chronic_conditions", "current_medications", mode="before")
    @classmethod
    def normalize_list_items(cls, v: list) -> list[str]:
        """Normalize: lowercase, strip whitespace, remove empty/duplicate items."""
        if not v:
            return []
        seen = set()
        result = []
        for item in v:
            cleaned = str(item).strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    # ── Computed Risk Properties ──

    HIGH_RISK_CONDITIONS: ClassVar[frozenset] = frozenset({
        "diabetes", "heart disease", "hypertension", "cancer",
        "immunocompromised", "hiv", "aids", "copd", "kidney disease",
        "liver disease", "autoimmune", "heart failure", "stroke",
        "chronic kidney disease", "blood thinners",
    })

    COMMON_DRUG_ALLERGIES: ClassVar[frozenset] = frozenset({
        "penicillin", "sulfa", "aspirin", "ibuprofen", "nsaid",
        "codeine", "morphine", "latex", "cephalosporin",
    })

    @property
    def has_high_risk_conditions(self) -> bool:
        """True if any chronic condition is in the high-risk set."""
        return bool(
            set(c.lower() for c in self.chronic_conditions) & self.HIGH_RISK_CONDITIONS
        )

    @property
    def has_drug_allergies(self) -> bool:
        """True if any allergy is a common drug allergy."""
        return bool(
            set(a.lower() for a in self.allergies) & self.COMMON_DRUG_ALLERGIES
        )

    @property
    def is_on_blood_thinners(self) -> bool:
        """True if patient is on anticoagulant medications.
        Uses substring matching to handle medication strings with dosages (e.g. 'warfarin 5mg').
        """
        thinners = {"warfarin", "coumadin", "xarelto", "eliquis", "pradaxa",
                    "heparin", "aspirin", "plavix", "clopidogrel"}
        meds_lower = [m.lower() for m in self.current_medications]
        return any(
            thinner in med
            for med in meds_lower
            for thinner in thinners
        )
