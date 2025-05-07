import asyncio

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits

from multi_mcp.mermaid_diagrams import invalid_mermaid_diagram, valid_mermaid_diagram

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
        "run_server.py",
        "stdio",
    ],
)
mermaid_server = MCPServerStdio(
    command="npx",
    args=[
        "-y",
        "@rtuin/mcp-mermaid-validator@latest",
    ],
)
# Create Agent with MCP servers
agent = Agent(
    "gemini-2.5-pro-preview-03-25",
    # "openai:o4-mini",
    mcp_servers=[local_server, mermaid_server],
)
Agent.instrument_all()


async def main(query: str = "Hi!", request_limit: int = 5) -> None:
    """
    Main function to run the agent

    Args:
        query (str): The query to run the agent with
        request_limit (int): The number of requests to make to the MCP servers
    """
    # Set a request limit for LLM calls
    usage_limits = UsageLimits(request_limit=request_limit)

    # Invoke the agent with the usage limits
    async with agent.run_mcp_servers():
        result = await agent.run(query, usage_limits=usage_limits)
    # print(result.output)
    return result

    # Evals
    # 1. Did it use both MCP tools?
    # 2. Is the mermaid diagram valid?
    # 3. Is the mermaid diagram close to the ground truth?


if __name__ == "__main__":
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {invalid_mermaid_diagram}. Return only the fixed mermaid diagram between backticks."
    asyncio.run(main(query))
