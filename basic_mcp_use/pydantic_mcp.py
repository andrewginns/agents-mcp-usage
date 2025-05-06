import asyncio
import os

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

load_dotenv()

# Configure logging to logfire if LOGFIRE_TOKEN is set in environment
logfire.configure(send_to_logfire="if-token-present", service_name="pydantic-basic-mcp")
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

server = MCPServerStdio(
    command="uv",
    args=[
        "run",
        "run_server.py",
        "stdio",
    ],
)
agent = Agent("gemini-2.5-pro-preview-03-25", mcp_servers=[server])
Agent.instrument_all()


async def main(query: str = "Greet Andrew and give him the current time") -> None:
    """
    Main function to run the agent

    Args:
        query (str): The query to run the agent with
    """
    async with agent.run_mcp_servers():
        result = await agent.run(query)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
