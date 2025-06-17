import asyncio

import logfire
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv

from agents_mcp_usage.utils import get_mcp_server_path

load_dotenv()

# Configure Logfire
logfire.configure(send_to_logfire="if-token-present", service_name="oai-basic-mcp")
logfire.instrument_mcp()
logfire.instrument_openai_agents()


async def main(query: str = "Greet Andrew and give him the current time") -> None:
    """Runs the OpenAI agent with a given query.

    This function creates an MCP server, initializes an OpenAI agent with the
    server, and runs the agent with the provided query.

    Args:
        query: The query to run the agent with.
    """
    # Create and use the MCP server in an async context
    async with MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", str(get_mcp_server_path("example_server.py")), "stdio"],
        }
    ) as server:
        # Initialise the agent with the server
        agent = Agent(
            name="MCP agent",
            model="o4-mini",
            mcp_servers=[server],
        )

        result = await Runner.run(
            starting_agent=agent,
            input=query,
        )

        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
