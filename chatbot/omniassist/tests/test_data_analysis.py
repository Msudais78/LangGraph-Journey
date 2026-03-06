"""Tests for data analysis subgraph."""

from src.graphs.data_analysis import build_data_analysis_graph, graph


def test_data_analysis_compiles():
    builder = build_data_analysis_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_data_analysis_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["parse_request", "statistics", "visualization", "summary", "results_merger"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"
