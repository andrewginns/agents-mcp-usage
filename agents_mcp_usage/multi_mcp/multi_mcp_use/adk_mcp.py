"""Advanced multi-MCP server integration using Google's ADK.

This module demonstrates sophisticated ADK (Agent Development Kit) patterns including:
- Multi-MCP server coordination with parallel connections
- Latest gemini-2.0-flash model integration
- Advanced async connection management with unified exit stacks
- Comprehensive error handling and recovery
- RunConfig for request limiting and control
- Enhanced event tracking and metrics collection
- Graceful resource cleanup and Logfire instrumentation

Compatible with ADK v1.3.0+ and showcases production-ready patterns
for complex agent architectures involving multiple tool sources.
"""

import asyncio
import os
import time

import logfire
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
    StdioConnectionParams,
)
from google.genai import types

from agents_mcp_usage.multi_mcp.mermaid_diagrams import invalid_mermaid_diagram_easy

load_dotenv()

# Set API key for Google AI API from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Configure logging if LOGFIRE_TOKEN is set
logfire.configure(send_to_logfire="if-token-present", service_name="adk-multi-mcp")
logfire.instrument_mcp()


async def get_tools_async() -> tuple[list, list]:
    """Initializes connections to MCP servers and returns their tools.

    This function connects to the local example server and the mermaid
    validator server, and returns the combined tools and a combined exit
    stack for cleanup.

    Returns:
        A tuple containing the list of all tools and the list of toolsets for cleanup.
    """
    print("Connecting to MCP servers...")

    # Keep track of toolsets for cleanup (ADK v1.3.0+ API)
    toolsets = []

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

    # Connect to local python MCP server (ADK v1.3.0+ API)
    local_connection = StdioConnectionParams(server_params=local_server)
    local_toolset = MCPToolset(connection_params=local_connection)
    local_tools = await local_toolset.get_tools()
    toolsets.append(local_toolset)
    print(f"Connected to local python MCP server. Found {len(local_tools)} tools.")

    # Connect to npx mermaid MCP server (ADK v1.3.0+ API)
    mermaid_connection = StdioConnectionParams(server_params=mermaid_server)
    mermaid_toolset = MCPToolset(connection_params=mermaid_connection)
    mermaid_tools = await mermaid_toolset.get_tools()
    toolsets.append(mermaid_toolset)
    print(f"Connected to npx mermaid MCP server. Found {len(mermaid_tools)} tools.")

    # Combine tools from both servers
    all_tools = local_tools + mermaid_tools
    print(f"Total tools available: {len(all_tools)}")

    return all_tools, toolsets


async def main(query: str = "Hi!", request_limit: int = 5) -> None:
    """Runs the agent with a given query and request limit using modern ADK patterns.

    This function initialises the tools, creates an agent, and runs it with
    the provided query and request limit. It also handles the cleanup of
    the MCP server connections and Logfire.

    Args:
        query: The query to run the agent with.
        request_limit: The maximum number of LLM calls allowed.
    """
    toolsets = []
    try:
        # Get tools from MCP servers
        tools, toolsets = await get_tools_async()

        # Create agent
        agent = LlmAgent(
            model="gemini-2.0-flash",
            name="multi_mcp_adk",
            tools=tools,
        )

        # Create async session service
        session_service = InMemorySessionService()
        session = await session_service.create_session(
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

    except Exception as e:
        print(f"Error during agent execution: {e}")
        print(f"Error type: {type(e).__name__}")
        raise
    finally:
        # Clean up MCP toolsets to prevent asyncio shutdown errors
        print("Cleaning up MCP connections...")
        for i, toolset in enumerate(toolsets):
            try:
                await toolset.close()
                print(f"Toolset {i + 1} closed successfully.")
            except asyncio.CancelledError:
                print(f"Toolset {i + 1} cleanup cancelled - this is normal")
            except Exception as e:
                print(f"Warning during cleanup of toolset {i + 1}: {e}")
        print("MCP cleanup completed.")


if __name__ == "__main__":
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {invalid_mermaid_diagram_easy}. Return only the fixed mermaid diagram between backticks."
    asyncio.run(main(query))
