"""Calculator tool."""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 2', '(3 * 4) / 2'
    """
    try:
        import math
        allowed_names = {"__builtins__": {}}
        allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
        result = eval(expression, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
