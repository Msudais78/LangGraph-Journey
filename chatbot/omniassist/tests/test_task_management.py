"""Tests for the Task Management subgraph."""

from src.graphs.task_management import build_task_graph, graph


def test_task_graph_compiles():
    builder = build_task_graph()
    compiled = builder.compile()
    assert compiled is not None


def test_task_graph_has_expected_nodes():
    nodes = list(graph.get_graph().nodes)
    expected = ["classify_action", "find_related", "confirmation_gate", "execute_task_action"]
    for node_name in expected:
        assert node_name in nodes, f"Missing node: {node_name}"
