from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of the two numbers.
    """
    return a + b


# Add a tool to get the current time
@mcp.tool()
def get_current_time() -> str:
    """Gets the current time.

    Returns:
        The current time in 'YYYY-MM-DD HH:MM:SS' format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Gets a personalised greeting.

    Args:
        name: The name to include in the greeting.

    Returns:
        A personalised greeting string.
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()
