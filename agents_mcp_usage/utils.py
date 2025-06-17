"""Utility functions and constants for the agents-mcp-usage project."""

from pathlib import Path

# Get the project root directory (where pyproject.toml is located)
# This is 2 levels up from this utils.py file: agents_mcp_usage/utils.py -> agents-mcp-usage/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def get_project_root() -> Path:
    """Get the absolute path to the project root directory.

    Returns:
        Path: The absolute path to the project root where pyproject.toml is located.
    """
    return PROJECT_ROOT


def get_mcp_server_path(server_name: str) -> Path:
    """Get the absolute path to an MCP server file.

    Args:
        server_name: The name of the MCP server file (e.g., "example_server.py")

    Returns:
        Path: The absolute path to the MCP server file.
    """
    return PROJECT_ROOT / "mcp_servers" / server_name
