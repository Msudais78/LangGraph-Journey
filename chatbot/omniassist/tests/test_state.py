"""Tests for state schemas and reducers."""

import pytest
from src.state import append_reducer
from src.state.main_state import OmniAssistState, ValidatedInput, OmniAssistInput, OmniAssistOutput
from src.state.context import OmniAssistContext
from src.state.research_state import ResearchState
from src.state.writing_state import WritingState
from src.state.task_state import TaskState
from src.state.code_state import CodeState
from src.state.data_state import DataState


def test_append_reducer_extends_list():
    assert append_reducer([1, 2], [3, 4]) == [1, 2, 3, 4]


def test_append_reducer_appends_single():
    assert append_reducer([1, 2], 3) == [1, 2, 3]


def test_append_reducer_handles_none():
    assert append_reducer(None, [1, 2]) == [1, 2]


def test_validated_input_accepts_valid():
    v = ValidatedInput(messages=[{"role": "user", "content": "hi"}], user_id="user123")
    assert v.user_id == "user123"


def test_validated_input_rejects_empty_user():
    with pytest.raises(Exception):
        ValidatedInput(messages=[], user_id="   ")


def test_state_schemas_are_typed_dicts():
    for schema in [OmniAssistState, OmniAssistInput, OmniAssistOutput,
                   ResearchState, WritingState, TaskState, CodeState, DataState]:
        assert hasattr(schema, "__annotations__"), f"{schema.__name__} missing annotations"


def test_context_schema_has_expected_keys():
    expected_keys = {"model_name", "temperature", "persona", "max_research_sources",
                     "code_execution_enabled", "language"}
    assert expected_keys == set(OmniAssistContext.__annotations__.keys())
