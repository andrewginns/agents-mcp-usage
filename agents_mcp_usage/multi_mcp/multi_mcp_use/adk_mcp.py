"""Advanced multi-MCP server integration using Google's ADK.

This module demonstrates sophisticated ADK (Agent Development Kit) patterns including:
- Multi-MCP server coordination with parallel connections
- Latest gemini-2.0-flash model integration
- Advanced async connection management with unified exit stacks
- Comprehensive error handling and recovery
- RunConfig for request limiting and control
- Enhanced event tracking and metrics collection
- Graceful resource cleanup and Logfire instrumentation
- Web UI integration via ADK's callback system

Compatible with ADK v1.3.0+ and showcases production-ready patterns
for complex agent architectures involving multiple tool sources.

This module provides an ADK agent that can be used both in the ADK web UI
and directly from the command line. The agent uses multiple MCP servers
to access different external functionalities.
"""

import asyncio
import os
from typing import List, Tuple, Any

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
from agents_mcp_usage.utils import get_mcp_server_path

load_dotenv()

# Set API key for Google AI API from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Configure logging if LOGFIRE_TOKEN is set
logfire.configure(send_to_logfire="if-token-present", service_name="adk-multi-mcp")
logfire.instrument_mcp()

# Global variable to store toolset instances for cleanup
_ACTIVE_TOOLSETS = []

# Flag to track if tools have been attached
TOOLS_ATTACHED = False


async def get_tools_async() -> Tuple[List[Any], List[MCPToolset]]:
    """Initializes connections to MCP servers and returns their tools.

    This function connects to the local example server and the mermaid
    validator server, and returns the combined tools and a list of toolsets
    for cleanup.

    Returns:
        A tuple containing the list of all tools and the list of toolsets for cleanup.
    """
    global _ACTIVE_TOOLSETS
    print("Connecting to MCP servers...")

    # Set up MCP server connections
    local_server = StdioServerParameters(
        command="uv",
        args=[
            "run",
            str(get_mcp_server_path("example_server.py")),
            "stdio",
        ],
    )

    mermaid_server = StdioServerParameters(
        command="uv",
        args=[
            "run",
            str(get_mcp_server_path("mermaid_validator.py")),
        ],
    )

    local_toolset = None
    mermaid_toolset = None
    toolsets = []

    try:
        # Connect to local python MCP server
        local_connection = StdioConnectionParams(server_params=local_server)
        local_toolset = MCPToolset(connection_params=local_connection)
        local_tools = await local_toolset.get_tools()
        toolsets.append(local_toolset)
        _ACTIVE_TOOLSETS.append(local_toolset)
        print(f"Connected to local python MCP server. Found {len(local_tools)} tools.")

        # Connect to mermaid MCP server
        mermaid_connection = StdioConnectionParams(server_params=mermaid_server)
        mermaid_toolset = MCPToolset(connection_params=mermaid_connection)
        mermaid_tools = await mermaid_toolset.get_tools()
        toolsets.append(mermaid_toolset)
        _ACTIVE_TOOLSETS.append(mermaid_toolset)
        print(f"Connected to mermaid MCP server. Found {len(mermaid_tools)} tools.")

        # Combine tools from both servers
        all_tools = local_tools + mermaid_tools
        print(f"Total tools available: {len(all_tools)}")

        return all_tools, toolsets

    except Exception as e:
        # Clean up in case of initialisation error
        if local_toolset in toolsets:
            try:
                await local_toolset.close()
                toolsets.remove(local_toolset)
                if local_toolset in _ACTIVE_TOOLSETS:
                    _ACTIVE_TOOLSETS.remove(local_toolset)
            except Exception:
                pass

        if mermaid_toolset in toolsets:
            try:
                await mermaid_toolset.close()
                toolsets.remove(mermaid_toolset)
                if mermaid_toolset in _ACTIVE_TOOLSETS:
                    _ACTIVE_TOOLSETS.remove(mermaid_toolset)
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
    name="multi_mcp_adk_assistant",
    instruction="You are an assistant that uses multiple MCP servers to help users. You have access to both a Python MCP server with time tools and a Mermaid diagram validator.",
    before_agent_callback=attach_tools_callback,  # This ensures tools are attached
)


# Function to dynamically attach tools to the agent
async def ensure_tools_attached():
    """Ensures that tools are attached to the agent before it's used."""
    global TOOLS_ATTACHED

    if not TOOLS_ATTACHED:
        try:
            tools, _ = await get_tools_async()
            print(f"✓ Connected to MCP servers. Found {len(tools)} tools.")
            # Update the agent's tools
            root_agent.tools = tools
            TOOLS_ATTACHED = True
        except Exception as e:
            print(f"Error attaching MCP tools: {e}")
            # Set empty tools to avoid errors
            root_agent.tools = []


async def main(query: str = "Hi!", request_limit: int = 5) -> None:
    """Runs the agent with a given query and request limit using modern ADK patterns.

    This function initialises the tools, creates an agent, and runs it with
    the provided query and request limit. It also handles the cleanup of
    the MCP server connections and Logfire.

    Args:
        query: The query to run the agent with.
        request_limit: The maximum number of LLM calls allowed.
    """
    try:
        # Ensure tools are attached to the agent
        await ensure_tools_attached()

        # Create async session service
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="multi_mcp_adk",
            user_id="andrewginns",
        )

        # Create a RunConfig with a limit for LLM calls (500 is the default)
        run_config = RunConfig(max_llm_calls=request_limit)

        # Create runner using the globally defined agent
        runner = Runner(
            app_name="multi_mcp_adk",
            agent=root_agent,
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
        await cleanup_toolsets()
        print("MCP cleanup completed.")


if __name__ == "__main__":
    query = f"Add the current time and fix the mermaid diagram syntax using the validator: {invalid_mermaid_diagram_easy}. Return only the fixed mermaid diagram between backticks."
    asyncio.run(main(query))
