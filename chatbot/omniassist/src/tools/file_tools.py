"""File operation tools."""

from langchain_core.tools import tool


@tool
def read_file(filepath: str) -> str:
    """Read contents of a file.

    Args:
        filepath: Path to the file to read
    """
    try:
        with open(filepath, "r") as f:
            content = f.read(10000)
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file.

    Args:
        filepath: Path to the file to write
        content: Content to write
    """
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"File written: {filepath} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"
