import asyncio
import contextlib
import os
import time

import logfire
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types

from agents_mcp_usage.multi_mcp.mermaid_diagrams import invalid_mermaid_diagram_easy

load_dotenv()

# Set API key for Google AI API from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Configure logging if LOGFIRE_TOKEN is set
logfire.configure(send_to_logfire="if-token-present", service_name="adk-multi-mcp")
logfire.instrument_mcp()


async def get_tools_async():
    """Initializes connections to MCP servers and returns their tools along with a combined exit stack."""
    print("Connecting to MCP servers...")

    # Create a single exit stack for all connections
    exit_stack = contextlib.AsyncExitStack()

    # Set up MCP server connections
    local_server = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "mcp_servers/example_server.py",
            "stdio",
        ],
    )

    mermaid_server = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "mcp_servers/mermaid_validator.py",
        ],
    )

    # Connect to local python MCP server
    local_toolset = await MCPToolset.from_server(connection_params=local_server)
    local_tools, local_stack = local_toolset
    # Register with the exit stack
    await exit_stack.enter_async_context(local_stack)
    print(f"Connected to local python MCP server. Found {len(local_tools)} tools.")

    # Connect to npx mermaid MCP server
    mermaid_toolset = await MCPToolset.from_server(connection_params=mermaid_server)
    mermaid_tools, mermaid_stack = mermaid_toolset
    # Register with the exit stack
    await exit_stack.enter_async_context(mermaid_stack)
    print(f"Connected to npx mermaid MCP server. Found {len(mermaid_tools)} tools.")

    # Combine tools from both servers
    all_tools = local_tools + mermaid_tools
    print(f"Total tools available: {len(all_tools)}")

    return all_tools, exit_stack


async def main(query: str = "Hi!", request_limit: int = 5) -> None:
    """
    Main function to run the agent

    Args:
        query (str): The query to run the agent with
        request_limit (int): The number of requests to make to the MCP servers
    """
    # Get tools from MCP servers
    tools, exit_stack = await get_tools_async()

    # Create agent with tools
    agent = LlmAgent(
        model="gemini-2.5-pro-preview-03-25",
        name="multi_mcp_adk",
        tools=tools,
    )

    # Create session service
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="multi_mcp_adk",
        user_id="andrewginns",
    )

    # Create a RunConfig with a limit for LLM calls (500 is the default)
    run_config = RunConfig(max_llm_calls=request_limit)

    # Create runner
    runner = Runner(
        app_name="multi_mcp_adk",
        agent=agent,
        session_service=session_service,
    )

    # Format the query as a message
    message = types.Content(parts=[types.Part(text=query)], role="user")

    # Run the agent
    events_async = runner.run_async(
        session_id=session.id,
        user_id="andrewginns",
        new_message=message,
        run_config=run_config,
    )

    async for event in events_async:
        print(f"Event received: {event}")

    # Properly close all MCP connections
    print("Closing MCP server connections...")
    await exit_stack.aclose()
    print("Cleanup complete.")

    # Give Logfire time to complete any pending exports
    print("Shutting down Logfire...")
    logfire.shutdown()
    # Small delay to ensure export completes
    time.sleep(0.5)
    print("Logfire shutdown complete.")


if __name__ == "__main__":
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {invalid_mermaid_diagram_easy}. Return only the fixed mermaid diagram between backticks."
    asyncio.run(main(query))
