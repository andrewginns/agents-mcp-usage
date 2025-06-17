import asyncio
from typing import Any

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits

from agents_mcp_usage.multi_mcp.mermaid_diagrams import invalid_mermaid_diagram_easy
from agents_mcp_usage.utils import get_mcp_server_path

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(send_to_logfire="if-token-present", service_name="pydantic-multi-mcp")
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

# Configure MCP servers
local_server = MCPServerStdio(
    command="uv",
    args=[
        "run",
        str(get_mcp_server_path("example_server.py")),
        "stdio",
    ],
)
mermaid_server = MCPServerStdio(
    command="uv",
    args=[
        "run",
        str(get_mcp_server_path("mermaid_validator.py")),
    ],
)
# Create Agent with MCP servers
agent = Agent(
    "gemini-2.5-pro-preview-06-05",
    # "openai:o4-mini",
    mcp_servers=[local_server, mermaid_server],
)
Agent.instrument_all()


async def main(query: str = "Hi!", request_limit: int = 5) -> Any:
    """Runs the Pydantic agent with a given query and request limit.

    This function invokes the Pydantic agent with the provided query and
    usage limits, and prints the output.

    Args:
        query: The query to run the agent with.
        request_limit: The number of requests to make to the MCP servers.

    Returns:
        The result of the agent run.
    """
    # Set a request limit for LLM calls
    usage_limits = UsageLimits(request_limit=request_limit)

    # Invoke the agent with the usage limits
    async with agent.run_mcp_servers():
        result = await agent.run(query, usage_limits=usage_limits)
    print(result.output)
    return result


if __name__ == "__main__":
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {invalid_mermaid_diagram_easy}. Return only the fixed mermaid diagram between backticks."
    asyncio.run(main(query))
