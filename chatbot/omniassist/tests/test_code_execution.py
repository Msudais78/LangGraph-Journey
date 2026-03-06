"""Tests for code execution."""


def test_execute_code_safe():
    from src.tools.code_runner import run_python_code
    result = run_python_code.invoke({"code": "print(2 + 2)"})
    assert "4" in result


def test_execute_code_error():
    from src.tools.code_runner import run_python_code
    result = run_python_code.invoke({"code": "raise ValueError('test')"})
    assert "ValueError" in result
