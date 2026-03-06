"""Tests for the Research subgraph."""

from src.graphs.research import build_research_graph, graph
from src.graphs.source_validator import build_source_validator


def test_research_graph_compiles():
    builder = build_research_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_research_graph_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["query_planner", "web_search", "knowledge_base",
                "academic_search", "results_aggregator", "synthesis", "quality_check"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"


def test_source_validator_compiles():
    builder = build_source_validator()
    compiled = builder.compile()
    assert compiled is not None
