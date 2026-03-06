"""
tests/test_graph.py
===================
Graph compilation test — verifies the LangGraph compiles correctly.
Does NOT make API calls.
"""

import pytest
from langgraph.graph import StateGraph


def test_graph_compiles():
    """Graph should compile without errors."""
    from medical_symptom_checker.graph import build_medical_checker_graph
    graph = build_medical_checker_graph()
    assert graph is not None


def test_graph_has_nodes():
    """Graph should include all required nodes."""
    from medical_symptom_checker.graph import build_medical_checker_graph
    graph = build_medical_checker_graph()
    graph_dict = graph.get_graph()
    node_names = {n for n in graph_dict.nodes}

    required_nodes = {
        "greeting", "patient_profile", "symptom_intake",
        "follow_up", "red_flag_check", "medical_history",
        "severity_classifier", "emergency_guidance", "hospital_finder",
        "doctor_recommendation", "specialist_mapper", "home_remedy",
        "lifestyle_prevention", "drug_interaction_check",
        "mental_health_screen", "summary_report", "feedback_followup",
    }

    for node in required_nodes:
        assert node in node_names, f"Missing node: {node}"


def test_initial_state_factory():
    """create_initial_state should return a properly initialized dict."""
    from medical_symptom_checker.models.state import create_initial_state
    state = create_initial_state()
    assert isinstance(state, dict)
    assert "messages" in state
    assert "session_id" in state
    assert state["loop_count"] == 0
    assert state["max_loops"] == 5
