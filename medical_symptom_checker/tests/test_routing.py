"""
tests/test_routing.py
=====================
Unit tests for all routing edge functions.
No API calls — pure logic tests.
"""

import pytest
from medical_symptom_checker.routing.edges import (
    should_continue_followup,
    route_after_red_flag,
    route_by_severity,
    route_after_home_remedy,
    route_after_emergency,
)
from medical_symptom_checker.models.state import create_initial_state
from medical_symptom_checker.models.assessment import SeverityAssessment
from medical_symptom_checker.models.patient import PatientProfile


# ── should_continue_followup ──────────────────────────────────────

class TestFollowUpRouting:
    def test_routes_to_followup_when_needed(self):
        state = create_initial_state(needs_more_info=True, loop_count=0, max_loops=5)
        assert should_continue_followup(state) == "follow_up"

    def test_routes_to_red_flag_when_complete(self):
        state = create_initial_state(needs_more_info=False, loop_count=0, max_loops=5)
        assert should_continue_followup(state) == "red_flag_check"

    def test_routes_to_red_flag_at_loop_limit(self):
        state = create_initial_state(needs_more_info=True, loop_count=5, max_loops=5)
        assert should_continue_followup(state) == "red_flag_check"

    def test_routes_to_followup_below_limit(self):
        state = create_initial_state(needs_more_info=True, loop_count=3, max_loops=5)
        assert should_continue_followup(state) == "follow_up"


# ── route_after_red_flag ──────────────────────────────────────────

class TestRedFlagRouting:
    def test_routes_to_emergency_if_flag_triggered(self):
        state = create_initial_state(red_flag_triggered=True)
        assert route_after_red_flag(state) == "emergency_guidance"

    def test_routes_to_medical_history_if_no_flag(self):
        state = create_initial_state(red_flag_triggered=False)
        assert route_after_red_flag(state) == "medical_history"


# ── route_by_severity ────────────────────────────────────────────

class TestSeverityRouting:
    @pytest.mark.parametrize("level,expected", [
        ("emergency", "emergency_guidance"),
        ("severe", "emergency_guidance"),
        ("moderate", "doctor_recommendation"),
        ("mild", "home_remedy"),
    ])
    def test_all_severity_routes(self, level, expected):
        state = create_initial_state(
            severity_assessment=SeverityAssessment(level=level)
        )
        assert route_by_severity(state) == expected


# ── route_after_home_remedy ──────────────────────────────────────

class TestHomeRemedyRouting:
    def test_mild_goes_to_lifestyle(self):
        state = create_initial_state(
            severity_assessment=SeverityAssessment(level="mild")
        )
        assert route_after_home_remedy(state) == "lifestyle_prevention"

    def test_moderate_goes_to_drug_check(self):
        state = create_initial_state(
            severity_assessment=SeverityAssessment(level="moderate")
        )
        assert route_after_home_remedy(state) == "drug_interaction_check"


# ── route_after_emergency ────────────────────────────────────────

class TestEmergencyRouting:
    def test_routes_to_hospital_if_location_present(self):
        state = create_initial_state(
            patient_profile=PatientProfile(location="Dallas, TX")
        )
        assert route_after_emergency(state) == "hospital_finder"

    def test_routes_to_mental_health_if_no_location(self):
        state = create_initial_state(
            patient_profile=PatientProfile()  # No location
        )
        assert route_after_emergency(state) == "mental_health_screen"
