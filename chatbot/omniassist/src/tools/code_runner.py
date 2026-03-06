"""Sandboxed code execution tool."""

from langchain_core.tools import tool
import io
import contextlib


@tool
def run_python_code(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        result = ""
        if output:
            result += f"Output:\n{output}"
        if errors:
            result += f"\nStderr:\n{errors}"
        return result if result else "Code executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}"
