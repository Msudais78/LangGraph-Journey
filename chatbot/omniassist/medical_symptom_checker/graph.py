"""
graph.py
========
LangGraph Graph Assembly — wires all nodes and edges into a compiled StateGraph.

This is the main graph definition for the Medical Symptom Checker.

Flow:
                          ┌─────────────────┐
                          │   START         │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  greeting       │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ patient_profile │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ symptom_intake  │◄────────┐
                          └────────┬────────┘         │
                                   │                  │
              ┌────────────────────▼──────────────┐   │
              │  should_continue_followup()        │   │
              └────────────┬──────────────────────┘   │
                           │ needs_more_info           │
                   YES ────┤                           │
                           │                           │
              ┌────────────▼──────────────┐            │
              │      follow_up            │────────────┘
              └────────────────────────────┘
                           │ NO (enough info)
                  ┌────────▼────────┐
                  │  red_flag_check │
                  └────────┬────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │       route_after_red_flag()       │
         └────────────┬──────────────────────┘
                      │
         ┌────────────┴─────────────────┐
         │ EMERGENCY                    │ NO FLAGS
         ▼                              ▼
   emergency_guidance         medical_history
         │                              │
         │ route_after_emergency()      │
         ├─────────────────┐    ┌───────▼────────┐
         │ has location    │    │severity_classify│
         ▼                 │    └───────┬─────────┘
  hospital_finder          │            │
         │                 │   route_by_severity()
         └────────┬────────┘    ┌───────┴──────────────┐
                  │             │                       │
                  │         moderate                 mild
                  │             │                       │
                  │    doctor_recommendation     home_remedy
                  │             │                       │
                  │     specialist_mapper        lifestyle_prevention
                  │             │                       │
                  └─────────────┴───────────────────────┘
                                │
                        ┌───────▼────────┐
                        │ drug_interact  │
                        └───────┬────────┘
                                │
                        ┌───────▼────────┐
                        │ mental_health  │
                        └───────┬────────┘
                                │
                        ┌───────▼────────┐
                        │ summary_report │
                        └───────┬────────┘
                                │
                        ┌───────▼────────┐
                        │ feedback/fup   │
                        └───────┬────────┘
                                │
                             [END]
"""

from langgraph.graph import StateGraph, END

from medical_symptom_checker.core.logger import get_logger
from medical_symptom_checker.models.state import MedicalCheckerState

# ── Import all nodes ──────────────────────────────────────────────
from medical_symptom_checker.nodes.greeting import greeting_disclaimer_node
from medical_symptom_checker.nodes.patient_profile import patient_profile_node
from medical_symptom_checker.nodes.symptom_intake import symptom_intake_node
from medical_symptom_checker.nodes.follow_up import follow_up_question_node
from medical_symptom_checker.nodes.red_flag import red_flag_detector_node
from medical_symptom_checker.nodes.medical_history import medical_history_node
from medical_symptom_checker.nodes.severity import severity_classifier_node
from medical_symptom_checker.nodes.emergency import emergency_guidance_node, nearest_hospital_node
from medical_symptom_checker.nodes.moderate import doctor_visit_recommender_node, specialist_mapper_node
from medical_symptom_checker.nodes.mild import home_remedy_node, lifestyle_prevention_node
from medical_symptom_checker.nodes.shared import (
    drug_interaction_node,
    mental_health_screening_node,
    summary_report_node,
    feedback_followup_node,
)

# ── Import routing functions ──────────────────────────────────────
from medical_symptom_checker.routing.edges import (
    should_continue_followup,
    route_after_red_flag,
    route_by_severity,
    route_after_home_remedy,
    route_after_emergency,
)

logger = get_logger(__name__)


def build_medical_checker_graph() -> StateGraph:
    """
    Build and compile the Medical Symptom Checker LangGraph.

    Returns:
        Compiled StateGraph ready to invoke.

    Usage:
        graph = build_medical_checker_graph()
        for event in graph.stream(initial_state):
            ...
    """
    logger.info("Building Medical Checker LangGraph...")

    # ── Initialize Graph ─────────────────────────────────────────
    workflow = StateGraph(MedicalCheckerState)

    # ── Register Nodes ───────────────────────────────────────────
    workflow.add_node("greeting", greeting_disclaimer_node)
    workflow.add_node("patient_profile", patient_profile_node)
    workflow.add_node("symptom_intake", symptom_intake_node)
    workflow.add_node("follow_up", follow_up_question_node)
    workflow.add_node("red_flag_check", red_flag_detector_node)
    workflow.add_node("medical_history", medical_history_node)
    workflow.add_node("severity_classifier", severity_classifier_node)
    workflow.add_node("emergency_guidance", emergency_guidance_node)
    workflow.add_node("hospital_finder", nearest_hospital_node)
    workflow.add_node("doctor_recommendation", doctor_visit_recommender_node)
    workflow.add_node("specialist_mapper", specialist_mapper_node)
    workflow.add_node("home_remedy", home_remedy_node)
    workflow.add_node("lifestyle_prevention", lifestyle_prevention_node)
    workflow.add_node("drug_interaction_check", drug_interaction_node)
    workflow.add_node("mental_health_screen", mental_health_screening_node)
    workflow.add_node("summary_report", summary_report_node)
    workflow.add_node("feedback_followup", feedback_followup_node)

    # ── Set Entry Point ──────────────────────────────────────────
    workflow.set_entry_point("greeting")

    # ── Linear Edges (guaranteed flow) ──────────────────────────
    workflow.add_edge("greeting", "patient_profile")
    workflow.add_edge("patient_profile", "symptom_intake")
    workflow.add_edge("follow_up", "symptom_intake")  # Follow-up answers → re-intake

    # ── Conditional: Follow-Up Loop ──────────────────────────────
    workflow.add_conditional_edges(
        "symptom_intake",
        should_continue_followup,
        {
            "follow_up": "follow_up",
            "red_flag_check": "red_flag_check",
        }
    )

    # ── Conditional: Red Flag Fast-Path ──────────────────────────
    workflow.add_conditional_edges(
        "red_flag_check",
        route_after_red_flag,
        {
            "emergency_guidance": "emergency_guidance",
            "medical_history": "medical_history",
        }
    )

    # ── Conditional: Post-Emergency (hospital search or skip) ────
    workflow.add_conditional_edges(
        "emergency_guidance",
        route_after_emergency,
        {
            "hospital_finder": "hospital_finder",
            "mental_health_screen": "mental_health_screen",
        }
    )

    # ── Linear: After hospital search → mental health ────────────
    workflow.add_edge("hospital_finder", "mental_health_screen")

    # ── Linear: Medical History → Severity Classification ────────
    workflow.add_edge("medical_history", "severity_classifier")

    # ── Conditional: Core Severity Router ───────────────────────
    workflow.add_conditional_edges(
        "severity_classifier",
        route_by_severity,
        {
            "emergency_guidance": "emergency_guidance",
            "doctor_recommendation": "doctor_recommendation",
            "home_remedy": "home_remedy",
        }
    )

    # ── Moderate Pathway ─────────────────────────────────────────
    workflow.add_edge("doctor_recommendation", "specialist_mapper")
    workflow.add_edge("specialist_mapper", "home_remedy")  # Moderate: brief home care too

    # ── Conditional: Post Home-Remedy ────────────────────────────
    workflow.add_conditional_edges(
        "home_remedy",
        route_after_home_remedy,
        {
            "lifestyle_prevention": "lifestyle_prevention",
            "drug_interaction_check": "drug_interaction_check",
        }
    )

    # ── Mild Pathway Tail ────────────────────────────────────────
    workflow.add_edge("lifestyle_prevention", "drug_interaction_check")

    # ── Shared Tail ──────────────────────────────────────────────
    workflow.add_edge("drug_interaction_check", "mental_health_screen")
    workflow.add_edge("mental_health_screen", "summary_report")
    workflow.add_edge("summary_report", "feedback_followup")
    workflow.add_edge("feedback_followup", END)

    # ── Compile ──────────────────────────────────────────────────
    graph = workflow.compile()
    logger.info("✅ Medical Checker graph compiled successfully!")
    return graph


# ── Module-level compiled graph instance ─────────────────────────
# Can be imported directly by other modules
medical_checker_graph = build_medical_checker_graph()
