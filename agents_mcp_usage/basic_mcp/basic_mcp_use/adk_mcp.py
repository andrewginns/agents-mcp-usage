"""ADK-based agent using MCP protocol for tool access.

This module provides an ADK agent that can be used both in the ADK web UI
and directly from the command line. The agent uses MCP tools to access
external functionality.
"""

import asyncio
import os
import logfire
from typing import List, Tuple, Any

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
    StdioConnectionParams,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents_mcp_usage.utils import get_mcp_server_path

load_dotenv()

# Set API key for Google AI API from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Configure logging if LOGFIRE_TOKEN is set
logfire.configure(send_to_logfire="if-token-present", service_name="adk-basic-mcp")
logfire.instrument_mcp()

# Global variable to store toolset instances for cleanup
_ACTIVE_TOOLSETS = []


# Initialize the MCP tools and agent for ADK web UI
async def init_mcp_toolset() -> Tuple[List[Any], List[MCPToolset]]:
    """Initialize and return the MCP tools and toolsets.

    Returns:
        Tuple[List[Any], List[MCPToolset]]: A tuple of (tools, toolsets)
    """
    global _ACTIVE_TOOLSETS

    # Use absolute path to MCP server based on project root
    mcp_server_path = get_mcp_server_path("example_server.py")

    server_params = StdioServerParameters(
        command="uv",
        args=["run", str(mcp_server_path), "stdio"],
    )
    connection_params = StdioConnectionParams(server_params=server_params)
    toolset = MCPToolset(connection_params=connection_params)

    try:
        tools = await toolset.get_tools()
        _ACTIVE_TOOLSETS.append(toolset)
        return tools, [toolset]
    except Exception as e:
        # Clean up in case of initialization error
        try:
            await toolset.close()
        except Exception:
            pass
        raise e


async def cleanup_toolsets():
    """Clean up any active MCP toolset connections."""
    global _ACTIVE_TOOLSETS

    for toolset in _ACTIVE_TOOLSETS:
        try:
            await toolset.close()
            print("MCP toolset connection closed.")
        except asyncio.CancelledError:
            print("MCP cleanup cancelled - this is normal")
        except Exception as e:
            print(f"Warning: Error during toolset cleanup: {e}")

    _ACTIVE_TOOLSETS = []


# Define a before_agent_callback to attach tools
async def attach_tools_callback(callback_context):
    """Callback to attach tools to the agent before it runs.

    Args:
        callback_context: The callback context from ADK.

    Returns:
        None: The callback doesn't modify the content.
    """
    await ensure_tools_attached()
    return None


# This is the agent that will be imported by the ADK web UI
root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="mcp_adk_assistant",
    instruction="You are an assistant that uses MCP tools to help users.",
    before_agent_callback=attach_tools_callback,  # This ensures tools are attached
)


# Flag to track if tools have been attached
TOOLS_ATTACHED = False


# Function to dynamically attach tools to the agent
async def ensure_tools_attached():
    """Ensures that tools are attached to the agent before it's used."""
    global TOOLS_ATTACHED

    if not TOOLS_ATTACHED:
        try:
            tools, _ = await init_mcp_toolset()
            print(f"✓ Connected to MCP server. Found {len(tools)} tools.")
            # Update the agent's tools
            root_agent.tools = tools
            TOOLS_ATTACHED = True
        except Exception as e:
            print(f"Error attaching MCP tools: {e}")
            # Set empty tools to avoid errors
            root_agent.tools = []


async def main(query: str = "Greet Andrew and give him the current time") -> None:
    """Runs the agent with a given query.

    This function sets up a runner for the agent and runs it with a specified query.
    It also handles the cleanup of the MCP server connection.

    Args:
        query: The query to run the agent with.
    """
    try:
        # Ensure tools are attached to the agent
        await ensure_tools_attached()

        # Set up session with async service
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="mcp_adk_app",
            user_id="aginns",
        )

        # Create the runner using the globally defined agent
        runner = Runner(
            app_name="mcp_adk_app",
            agent=root_agent,
            session_service=session_service,
        )

        # Format the query as content
        content = types.Content(role="user", parts=[types.Part(text=query)])

        print("Running agent...")
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )

        async for event in events_async:
            print(f"Event received: {event}")

    except Exception as e:
        print(f"Error during agent execution: {e}")
        print(f"Error type: {type(e).__name__}")
        raise
    finally:
        # Clean up MCP toolsets to prevent asyncio shutdown errors
        await cleanup_toolsets()
        print("Agent execution completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
