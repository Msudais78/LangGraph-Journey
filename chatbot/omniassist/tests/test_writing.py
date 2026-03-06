"""Tests for the Writing subgraph."""

from src.graphs.writing import build_writing_graph, graph


def test_writing_graph_compiles():
    builder = build_writing_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_writing_graph_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["analyze_request", "generate_draft", "quality_review", "human_review", "finalize"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"
