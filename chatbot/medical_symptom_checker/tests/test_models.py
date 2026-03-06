"""
tests/test_models.py
====================
Unit tests for all Pydantic models.
No API calls — these are pure data validation tests.
"""

import pytest
from medical_symptom_checker.models.patient import PatientProfile, MedicalHistory
from medical_symptom_checker.models.symptom import Symptom, VitalSigns
from medical_symptom_checker.models.assessment import SeverityAssessment, Recommendation
from medical_symptom_checker.models.mental_health import MentalHealthScreen
from medical_symptom_checker.models.state import create_initial_state


# ══════════════════════════════════════════════
#  PatientProfile Tests
# ══════════════════════════════════════════════

class TestPatientProfile:
    def test_defaults(self):
        p = PatientProfile()
        assert p.name == ""
        assert p.age == 0
        assert p.gender == ""

    def test_gender_normalization(self):
        assert PatientProfile(gender="M").gender == "male"
        assert PatientProfile(gender="female").gender == "female"
        assert PatientProfile(gender="nb").gender == "non-binary"
        assert PatientProfile(gender="MALE").gender == "male"

    def test_blood_type_normalization(self):
        assert PatientProfile(blood_type="a+").blood_type == "A+"
        assert PatientProfile(blood_type="O-").blood_type == "O-"
        assert PatientProfile(blood_type="invalid").blood_type == ""

    def test_age_validation_bounds(self):
        with pytest.raises(Exception):
            PatientProfile(age=-1)
        with pytest.raises(Exception):
            PatientProfile(age=200)

    def test_is_elderly(self):
        assert PatientProfile(age=65).is_elderly is True
        assert PatientProfile(age=64).is_elderly is False

    def test_is_pediatric(self):
        assert PatientProfile(age=12).is_pediatric is True
        assert PatientProfile(age=18).is_pediatric is False

    def test_is_high_risk_age(self):
        assert PatientProfile(age=70).is_high_risk_age is True
        assert PatientProfile(age=3).is_high_risk_age is True
        assert PatientProfile(age=30).is_high_risk_age is False

    def test_display_name(self):
        p = PatientProfile(name="Alice", age=30, gender="female")
        assert "Alice" in p.display_name
        assert "30" in p.display_name


class TestMedicalHistory:
    def test_defaults(self):
        h = MedicalHistory()
        assert h.chronic_conditions == []
        assert h.current_medications == []

    def test_normalize_list_items(self):
        h = MedicalHistory(chronic_conditions=["Diabetes", "DIABETES", "hypertension"])
        assert "diabetes" in h.chronic_conditions
        # Duplicates removed
        assert h.chronic_conditions.count("diabetes") == 1
        assert "hypertension" in h.chronic_conditions

    def test_has_high_risk_conditions(self):
        h = MedicalHistory(chronic_conditions=["diabetes"])
        assert h.has_high_risk_conditions is True

        h2 = MedicalHistory(chronic_conditions=["seasonal allergies"])
        assert h2.has_high_risk_conditions is False

    def test_has_drug_allergies(self):
        h = MedicalHistory(allergies=["Penicillin"])
        assert h.has_drug_allergies is True

    def test_is_on_blood_thinners(self):
        h = MedicalHistory(current_medications=["warfarin 5mg"])
        assert h.is_on_blood_thinners is True


# ══════════════════════════════════════════════
#  Symptom Tests
# ══════════════════════════════════════════════

class TestSymptom:
    def test_required_name(self):
        s = Symptom(name="headache")
        assert s.name == "headache"

    def test_name_normalization(self):
        s = Symptom(name="  HEADACHE  ")
        assert s.name == "headache"

    def test_onset_normalization(self):
        s = Symptom(name="pain", onset="sudden")
        assert s.onset == "sudden"
        s2 = Symptom(name="pain", onset="abrupt")
        assert s2.onset == "sudden"

    def test_is_acute(self):
        s = Symptom(name="chest pain", onset="sudden")
        assert s.is_acute is True

    def test_is_severe(self):
        s = Symptom(name="pain", intensity=8)
        assert s.is_severe is True
        s2 = Symptom(name="pain", intensity=6)
        assert s2.is_severe is False

    def test_intensity_bounds(self):
        with pytest.raises(Exception):
            Symptom(name="pain", intensity=11)
        with pytest.raises(Exception):
            Symptom(name="pain", intensity=-1)

    def test_clinical_summary(self):
        s = Symptom(name="headache", intensity=7, duration="2 days", onset="gradual")
        summary = s.clinical_summary
        assert "headache" in summary
        assert "7/10" in summary


class TestVitalSigns:
    def test_defaults_all_none(self):
        v = VitalSigns()
        assert v.temperature_f is None
        assert v.heart_rate_bpm is None

    def test_has_fever(self):
        v = VitalSigns(temperature_f=101.0)
        assert v.has_fever is True

    def test_no_fever(self):
        v = VitalSigns(temperature_f=98.6)
        assert v.has_fever is False

    def test_has_high_fever(self):
        v = VitalSigns(temperature_f=104.0)
        assert v.has_high_fever is True

    def test_low_oxygen(self):
        v = VitalSigns(oxygen_saturation=92.0)
        assert v.has_low_oxygen is True

    def test_normal_oxygen(self):
        v = VitalSigns(oxygen_saturation=98.0)
        assert v.has_low_oxygen is False

    def test_blood_pressure_valid(self):
        v = VitalSigns(blood_pressure="120/80")
        assert v.blood_pressure == "120/80"

    def test_blood_pressure_invalid(self):
        v = VitalSigns(blood_pressure="invalid")
        assert v.blood_pressure is None


# ══════════════════════════════════════════════
#  Assessment Tests
# ══════════════════════════════════════════════

class TestSeverityAssessment:
    def test_defaults(self):
        s = SeverityAssessment()
        assert s.level == "mild"
        assert s.confidence_score == 0.0

    def test_is_emergency(self):
        assert SeverityAssessment(level="emergency").is_emergency is True
        assert SeverityAssessment(level="severe").is_emergency is True
        assert SeverityAssessment(level="moderate").is_emergency is False
        assert SeverityAssessment(level="mild").is_emergency is False

    def test_has_red_flags(self):
        s = SeverityAssessment(red_flags_detected=["🚩 CHEST PAIN"])
        assert s.has_red_flags is True

    def test_severity_emoji(self):
        assert SeverityAssessment(level="mild").severity_emoji == "🟢"
        assert SeverityAssessment(level="moderate").severity_emoji == "🟡"
        assert SeverityAssessment(level="severe").severity_emoji == "🔴"
        assert SeverityAssessment(level="emergency").severity_emoji == "🚨"

    def test_top_conditions(self):
        s = SeverityAssessment(possible_conditions=[
            {"name": "COVID-19", "probability": 0.7},
            {"name": "Flu", "probability": 0.2},
        ])
        top = s.top_conditions
        assert len(top) == 2
        assert "COVID-19" in top[0]


# ══════════════════════════════════════════════
#  State Factory Tests
# ══════════════════════════════════════════════

class TestCreateInitialState:
    def test_returns_dict(self):
        state = create_initial_state()
        assert isinstance(state, dict)

    def test_all_required_fields(self):
        state = create_initial_state()
        required = [
            "messages", "current_step", "patient_profile", "medical_history",
            "vital_signs", "symptoms", "loop_count", "max_loops",
            "red_flag_triggered", "needs_more_info", "final_report",
            "session_id", "timestamp",
        ]
        for field in required:
            assert field in state, f"Missing field: {field}"

    def test_override_works(self):
        state = create_initial_state(loop_count=3, max_loops=10)
        assert state["loop_count"] == 3
        assert state["max_loops"] == 10

    def test_session_id_unique(self):
        s1 = create_initial_state()
        s2 = create_initial_state()
        assert s1["session_id"] != s2["session_id"]

    def test_defaults_safe(self):
        state = create_initial_state()
        assert state["red_flag_triggered"] is False
        assert state["loop_count"] == 0
        assert state["disclaimer_acknowledged"] is False
