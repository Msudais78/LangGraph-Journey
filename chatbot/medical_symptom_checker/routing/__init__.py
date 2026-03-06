"""Routing/edge functions."""
from medical_symptom_checker.routing.edges import (
    should_continue_followup,
    route_after_red_flag,
    route_by_severity,
    route_after_home_remedy,
    route_after_emergency,
)

__all__ = [
    "should_continue_followup",
    "route_after_red_flag",
    "route_by_severity",
    "route_after_home_remedy",
    "route_after_emergency",
]
